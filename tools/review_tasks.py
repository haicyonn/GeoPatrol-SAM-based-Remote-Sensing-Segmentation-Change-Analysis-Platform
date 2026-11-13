"""Small, persistent review task service. No extra agent framework or writes to assets."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import queue
import re
import secrets
import sqlite3
import threading
import time

from review_planning import Snapshot, compile_request, select_batch, digest, FAMILY_LABELS, amount, NUM
from inspection_knowledge import public_citations


class ReviewError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Cancelled(Exception):
    pass


CHECKS = {
    'fragmentation': '对照双时相与叠加图，逐一核对零散小区域是否为真实变化，避免直接删去小目标。',
    'holes': '对照双时相影像核对变化范围内部空洞，确认是真实背景还是遗漏；不要直接全部填满。',
    'border': '查看相邻切片或原始大图，核对边缘变化是否跨越切片边界。',
    'tiny': '放大原图核对极小响应，区分细小真实变化与噪声。'}


def extract_plan(output):
    if isinstance(output, str): output = {'text': output, 'finishReason': 'stop'}
    if not isinstance(output, dict) or output.get('finishReason') in {'length', 'cancelled', 'timeout'}:
        raise ValueError('model output did not complete')
    text = str(output.get('text') or '').strip()
    if '</think>' in text: text = text.rsplit('</think>', 1)[1].strip()
    elif '<think>' in text: raise ValueError('incomplete reasoning')
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
    if len(text) > 20000: raise ValueError('oversized model plan')
    return json.loads(text)


class ReviewTasks:
    def __init__(self, source_path, db_path, *, generate=None, retrieve=None):
        self.source_path = Path(source_path)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.generate, self.retrieve = generate, retrieve
        self._source = Snapshot(self.source_path)
        self._signature = self._stat()
        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._cancel = {}
        self._closed = False
        with self.db() as db:
            db.executescript('''
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS review_jobs(
                job_id TEXT PRIMARY KEY,owner TEXT NOT NULL,state TEXT NOT NULL,stage TEXT,
                payload TEXT NOT NULL,result TEXT,error TEXT,created REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS review_plans(
                task_id TEXT NOT NULL,version INTEGER NOT NULL,owner TEXT NOT NULL,
                source_version TEXT NOT NULL,result TEXT NOT NULL,created REAL NOT NULL,
                PRIMARY KEY(task_id,version));
              CREATE TABLE IF NOT EXISTS review_events(
                event_id INTEGER PRIMARY KEY,owner TEXT NOT NULL,request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,source_version TEXT NOT NULL,sample_id TEXT NOT NULL,
                facts_hash TEXT NOT NULL,note TEXT NOT NULL,actor TEXT NOT NULL,response TEXT NOT NULL,
                created REAL NOT NULL,UNIQUE(owner,request_id));
            ''')
            db.execute("UPDATE review_jobs SET state='interrupted',stage='服务已重启，请重新提交任务' WHERE state IN ('queued','running')")
        self._worker = threading.Thread(target=self._work, daemon=True, name='review-planning')
        self._worker.start()

    @contextmanager
    def db(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _stat(self):
        s = self.source_path.stat()
        return (s.st_mtime_ns, s.st_size)

    def source(self):
        with self._lock:
            signature = self._stat()
            if signature != self._signature:
                self._source = Snapshot(self.source_path); self._signature = signature
            return self._source

    @staticmethod
    def owner(payload):
        key = str(payload.get('sessionKey') or '')
        if not key or len(key) > 2000: raise ReviewError('复核任务需要有效会话。')
        return digest([str(payload.get('_authenticatedUser') or 'local'), key])

    def close(self):
        with self._lock:
            if self._closed: return
            self._closed = True
            for event in self._cancel.values(): event.set()
            self._queue.put(None)
        self._worker.join(timeout=3)

    def _previous(self, payload, owner):
        task_id = str(payload.get('reviewTaskId') or '')
        if not task_id: return None
        with self.db() as db:
            row = db.execute('SELECT result FROM review_plans WHERE task_id=? AND owner=? ORDER BY version DESC LIMIT 1', (task_id, owner)).fetchone()
        if row is None: raise ReviewError('找不到本会话中的复核清单。', 404)
        result = json.loads(row['result']); p = result['reviewPlan']
        if type(payload.get('reviewPlanVersion')) is not int or payload['reviewPlanVersion'] != p['version']:
            raise ReviewError('清单版本已变化，请使用最新一版。', 409)
        if p['sourceVersion'] != self.source().source_version:
            raise ReviewError('样本快照已更新，请重新生成清单。', 409)
        return {'plan': p['conditions'], 'rows': result['evidenceSamples'],
                'taskId': p['taskId'], 'version': p['version'], 'sourceVersion': p['sourceVersion'],
                'queryOrigin': result.get('queryOrigin')}

    def submit(self, payload):
        owner = self.owner(payload)
        if not isinstance(payload.get('question'), str) or not payload['question'].strip():
            raise ReviewError('请输入复核任务。')
        previous = self._previous(payload, owner)
        q = payload['question']
        revision = bool(previous and re.search(r'移出|换掉|替换|去掉|每类|至少|改成|改为|其他条件|保持|不限制|最低数量|重新选|全部重选', q))
        request = {k: copy.deepcopy(payload[k]) for k in ('question', 'region', 'sessionKey', '_authenticatedUser') if k in payload}
        request['_previous'] = previous if revision else None
        request['_sourceVersion'] = self.source().source_version
        with self._lock, self.db() as db:
            if self._closed: raise ReviewError('任务服务正在关闭。', 503)
            pending = db.execute("SELECT COUNT(*) FROM review_jobs WHERE state IN ('queued','running')").fetchone()[0]
            if pending >= 2: raise ReviewError('已有任务正在处理，请等待当前任务完成。', 429)
            job_id = secrets.token_urlsafe(18)
            db.execute('INSERT INTO review_jobs(job_id,owner,state,stage,payload,created) VALUES(?,?,?,?,?,?)',
                       (job_id, owner, 'queued', '等待处理', json.dumps(request, ensure_ascii=False), time.time()))
            event = threading.Event(); self._cancel[job_id] = event
        self._queue.put((job_id, owner, request, event))
        return {'reviewJob': True, 'jobId': job_id, 'state': 'queued', 'answer': '复核任务已提交，正在理解条件。'}

    def get(self, job_id, owner):
        with self.db() as db:
            row = db.execute('SELECT job_id,state,stage,result,error FROM review_jobs WHERE job_id=? AND owner=?', (str(job_id), owner)).fetchone()
        if row is None: raise ReviewError('任务不存在或不属于当前会话。', 404)
        result = json.loads(row['result']) if row['result'] else None
        if result and result['reviewPlan']['sourceVersion'] == self.source().source_version:
            confirmed = self.reviewed_ids()
            result['reviewConfirmedIds'] = [r['sample_id'] for r in result['evidenceSamples'] if r['sample_id'] in confirmed]
        return {'jobId': row['job_id'], 'state': row['state'], 'stage': row['stage'],
                'result': result, 'error': row['error']}

    def cancel(self, job_id, owner):
        state = self.get(job_id, owner)
        if state['state'] in {'queued', 'running'}:
            with self._lock:
                event = self._cancel.get(job_id)
                if event: event.set()
                with self.db() as db:
                    db.execute("UPDATE review_jobs SET state='cancelled',stage='排队任务已取消' WHERE job_id=? AND state='queued'", (job_id,))
            if state['state'] == 'running': self._stage(job_id, '正在取消；若模型正在生成，将在停止生成后结束')
        return {'ok': True, 'jobId': job_id}

    def reviewed_ids(self):
        version = self.source().source_version
        with self.db() as db:
            return {r[0] for r in db.execute('SELECT DISTINCT sample_id FROM review_events WHERE source_version=?', (version,))}

    def _review_revision(self, db, source_version):
        return db.execute('SELECT COALESCE(MAX(event_id),0) FROM review_events WHERE source_version=?', (source_version,)).fetchone()[0]

    def _stage(self, job_id, stage):
        with self.db() as db: db.execute('UPDATE review_jobs SET stage=? WHERE job_id=?', (stage, job_id))

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None: return
            job_id, owner, payload, event = item
            try:
                if event.is_set(): raise Cancelled()
                with self.db() as db: db.execute("UPDATE review_jobs SET state='running' WHERE job_id=?", (job_id,))
                result = self._execute(job_id, owner, payload, event)
                # The final commit is the cancellation boundary. A cancellation
                # arriving afterwards must not hide an already saved plan.
                with self.db() as db:
                    db.execute('UPDATE review_jobs SET state=?,stage=?,result=? WHERE job_id=?',
                               ('waiting_user' if result['needsClarification'] else 'completed', '条件需要补充' if result['needsClarification'] else '清单已生成', json.dumps(result, ensure_ascii=False), job_id))
            except Cancelled:
                with self.db() as db: db.execute("UPDATE review_jobs SET state='cancelled',stage='已取消；未生成新清单' WHERE job_id=?", (job_id,))
            except Exception as exc:
                with self.db() as db: db.execute("UPDATE review_jobs SET state='failed',stage='任务未完成',error=? WHERE job_id=?", (str(exc)[:600], job_id))
            finally:
                with self._lock: self._cancel.pop(job_id, None)

    def _prompt(self, q, source, base, previous):
        # The 7B model interprets fuzzy language only. Explicit numeric/scope
        # constraints are already compiled and verified outside the model.
        return ('将遥感复核要求中提到的质量问题归类。只输出JSON对象，格式如下：'
                '{"coverage":[],"unresolved":[]}。禁止其他字段。\n'
                '四个可选类别互不包含：\n'
                'fragmentation：变化区域是许多分散小块、破碎连通块或不规则碎片。\n'
                'holes：一个变化主体内部有缺失、缺口或漏掉的小块。\n'
                'border：变化被整张图像的外边缘截断。内部缺失不属于此类。\n'
                'tiny：整张图总变化面积很小；许多碎片不一定总面积小。\n'
                '只列用户实际描述的类别，不要为了全面而增加类别。'
                '未提及质量偏好时coverage为空；无法归类的偏好在unresolved引用用户原话。'
                '区域、数量、数值阈值、时间条件已有程序核验，不由你评估，也不要写入unresolved。\n'
                '例："主体中间漏掉一块" -> {"coverage":["holes"],"unresolved":[]}\n'
                '例："到处分散成许多不规则小块" -> {"coverage":["fragmentation"],"unresolved":[]}\n'
                '原有偏好：' + json.dumps(base['coverage'], ensure_ascii=False)
                + '\n用户要求：' + json.dumps(q, ensure_ascii=False)
                + '\n输出JSON：')

    def _execute(self, job_id, owner, payload, event):
        started = time.monotonic(); deadline = started + 180
        source = self.source()
        if source.source_version != payload['_sourceVersion']: raise ReviewError('数据源已更新，请重新提交。', 409)
        previous = payload.get('_previous')
        if previous:
            self._previous({'reviewTaskId': previous['taskId'], 'reviewPlanVersion': previous['version']}, owner)
        q = payload['question']
        region_id = str((payload.get('region') or {}).get('id') or '')
        plan = compile_request(q, region_id, source.regions, previous=previous)
        calls, usage, planner_status = 0, [], 'rules_only'
        trace = []
        self._stage(job_id, '理解复核条件')
        deterministic_revision = previous and re.search(r'第\s*' + NUM + r'\s*(?:张|个|幅).*(?:移出|换掉|替换|去掉)', q)
        preflight = select_batch(source.rows(plan['region_id']), plan, self.reviewed_ids())
        known_shortage = not plan['issues'] and not preflight['hardSatisfied']
        if self.generate and not plan['issues'] and not deterministic_revision and not known_shortage:
            prompt = self._prompt(q, source, plan, previous)
            error = None
            for attempt in range(2):
                if event.is_set(): raise Cancelled()
                self._stage(job_id, 'DeepSeek正在解析条件' if not attempt else 'DeepSeek正在修复任务格式')
                calls += 1
                try:
                    output = self.generate(prompt, max_tokens=1024, stop_event=event, deadline=deadline)
                    if event.is_set(): raise Cancelled()
                    if isinstance(output, dict): usage.append({k: output[k] for k in ('inputTokens', 'outputTokens', 'finishReason', 'latencyMs') if k in output})
                    candidate = extract_plan(output)
                    plan = compile_request(q, region_id, source.regions, previous=previous, model_plan=candidate)
                    planner_status = 'model' if not attempt else 'model_repaired'
                    error = None; break
                except Cancelled: raise
                except Exception as exc:
                    error = str(exc)[:150]
                    prompt = self._prompt(q, source, plan, previous) + '\n上次输出格式错误：' + error + '。请重新输出合法JSON；不要附加其他文字。'
            if error:
                planner_status = 'model_failed'
                plan['issues'].append('DeepSeek未能返回完整合法的任务计划，请改写需求后重试。')
        elif deterministic_revision: planner_status = 'deterministic_revision'
        elif known_shortage: planner_status = 'preflight_shortage'
        if event.is_set(): raise Cancelled()
        if time.monotonic() > deadline: raise ReviewError('任务超时，请缩小需求后重试。', 408)
        trace.append({'tool': 'compile_review_request', 'status': planner_status,
                      'summary': '解析复核条件；明确约束独立保留' + ('，纠正模型遗漏：' + '、'.join(plan['preservedConstraints']) if plan['preservedConstraints'] else '')})
        self._stage(job_id, '核对候选数量与问题类型覆盖')
        with self.db() as db: review_revision = self._review_revision(db, source.source_version)
        selection = select_batch(source.rows(plan['region_id']), plan, self.reviewed_ids())
        trace.append({'tool': 'select_review_batch', 'status': selection['status'],
                      'summary': f'候选{selection["candidateCount"]}个，选择{len(selection["rows"])}个；同时检查数量、排除条件和类型覆盖'})
        hits = []
        if self.retrieve and selection['rows']:
            self._stage(job_id, '补充对应质量问题的检查依据')
            try:
                query = ' '.join(FAMILY_LABELS[f] for f in plan['coverage']) + ' 人工复核操作步骤'
                hits = self.retrieve(query)
                trace.append({'tool': 'search_knowledge', 'status': 'ok' if hits else 'empty', 'summary': f'检索到{len(hits)}条检查依据'})
            except Exception as exc:
                trace.append({'tool': 'search_knowledge', 'status': 'unavailable', 'summary': type(exc).__name__})
        if event.is_set(): raise Cancelled()
        task_id = previous['taskId'] if previous else secrets.token_urlsafe(16)
        version = previous['version'] + 1 if previous else 1
        result = self._response(task_id, version, source, plan, selection, trace, calls, usage, planner_status, hits, review_revision)
        result['latencyMs'] = round((time.monotonic() - started) * 1000, 2)
        with self._lock, self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if event.is_set(): raise Cancelled()
            if self.source().source_version != source.source_version: raise ReviewError('数据源已更新，请重新提交。', 409)
            if self._review_revision(db, source.source_version) != review_revision: raise ReviewError('复核记录刚刚更新，请重新生成清单。', 409)
            latest = db.execute('SELECT MAX(version) FROM review_plans WHERE task_id=?', (task_id,)).fetchone()[0]
            if latest is not None and latest != version - 1: raise ReviewError('清单已被更新，请使用最新版本。', 409)
            db.execute('INSERT INTO review_plans VALUES(?,?,?,?,?,?)', (task_id, version, owner, source.source_version, json.dumps(result, ensure_ascii=False), time.time()))
        return result

    def _response(self, task_id, version, source, plan, selection, trace, calls, usage, status, hits, review_revision):
        rows = []
        for r in selection['rows']:
            r = copy.deepcopy(r)
            r['checklist'] = [CHECKS[f] for f in r['families'] if f in CHECKS] or ['对照前后时相与叠加图，核对变化主体和边界；当前没有明确的问题类型标记。']
            rows.append(r)
        coverage_text = '、'.join(FAMILY_LABELS[f] + f'{selection["coverage"].get(f, 0)}张' for f in plan['coverage']) or '按当前规则风险优先'
        intro = f'已生成第{version}版复核清单，共{len(rows)}张。' if selection['hardSatisfied'] else '这次需求尚未全部满足，清单仅供查看，不能当作已完成任务。'
        answer = intro + '\n覆盖：' + coverage_text + '。'
        if selection['issues']: answer += '\n' + '\n'.join(selection['issues'])
        if selection['softUnmet']: answer += '\n未覆盖的偏好：' + '、'.join(FAMILY_LABELS[f] for f in selection['softUnmet']) + '。'
        if status == 'rules_only': answer += '\n本次为规则解析，未调用DeepSeek。'
        elif status == 'preflight_shortage': answer += '\n已先用样本事实确认条件不足，本次未调用DeepSeek。'
        return {'answer': answer, 'answerScope': 'review_plan', 'engine': 'DeepSeek task planning + verified selection' if status in {'model', 'model_repaired'} else 'verified review selection',
                'mode': 'review_plan', 'dataSource': 'static_snapshot+local_reviews', 'grounded': bool(rows),
                'needsClarification': not selection['hardSatisfied'], 'llmCalls': calls, 'plannerStatus': status,
                'modelUsage': usage, 'toolTrace': trace, 'citations': public_citations(hits), 'evidenceSamples': rows,
                'reviewPlan': {'taskId': task_id, 'version': version, 'regionId': plan['region_id'],
                    'sourceVersion': source.source_version, 'reviewRevision': review_revision,
                    'conditions': plan, 'candidateCount': selection['candidateCount'], 'coverage': selection['coverage'],
                    'availableCoverage': selection['availableCoverage'], 'hardSatisfied': selection['hardSatisfied'],
                    'softUnmet': selection['softUnmet'], 'issues': selection['issues']}}

    def adopt_query(self,selection,payload):
        from query_review import adopt_query
        return adopt_query(self,selection,payload)

    def reference(self, payload):
        previous = self._previous(payload, self.owner(payload))
        if previous is None: raise ReviewError('请先生成复核清单。')
        match = re.search(r'第\s*(' + NUM + r')\s*(?:张|个|幅)', payload.get('question', ''))
        index = int(amount(match[1])) - 1 if match else -1
        if not 0 <= index < len(previous['rows']): raise ReviewError('序号不在指定版本的清单中。')
        row = previous['rows'][index]
        answer = f'{row["sample_id"]}：变化像素{row["changed_px"]}px，对象数量{row["object_count"]}个。'
        if re.search(r'为什么|依据|检查|怎么', payload.get('question', '')):
            answer += '\n' + '\n'.join(row.get('checklist', []))
        return {'answer': answer, 'answerScope': 'review_reference', 'evidenceSamples': [row],
                'reviewContext': {'taskId': previous['taskId'], 'version': previous['version']},
                'dataSource': 'static_snapshot+local_reviews', 'llmCalls': 0, 'toolTrace': [{'tool': 'resolve_review_reference', 'status': 'ok', 'summary': '按指定清单版本绑定样本'}]}

    def confirm(self, payload, owner):
        note = str(payload.get('note') or '').strip()
        request_id = str(payload.get('requestId') or '')
        if not 2 <= len(note) <= 1000 or not 1 <= len(request_id) <= 128:
            raise ReviewError('请填写人工核对结论，并提供有效保存请求。')
        request_hash = digest({k: payload.get(k) for k in ('reviewTaskId', 'reviewPlanVersion', 'sampleId', 'factsHash', 'note')})
        with self._lock, self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT request_hash,response FROM review_events WHERE owner=? AND request_id=?', (owner, request_id)).fetchone()
            if existing:
                if existing['request_hash'] != request_hash: raise ReviewError('该保存请求已用于另一条结论。', 409)
                return json.loads(existing['response'])
            previous = self._previous(payload, owner)
            if previous is None: raise ReviewError('请先生成清单。')
            sid = str(payload.get('sampleId') or '')
            row = next((r for r in previous['rows'] if r['sample_id'] == sid), None)
            if row is None or row['facts_hash'] != payload.get('factsHash'): raise ReviewError('样本或事实版本不匹配，请重新查看。', 409)
            fresh = next((r for r in self.source().rows(row['region_id']) if r['sample_id'] == sid), None)
            if fresh is None or fresh['facts_hash'] != row['facts_hash']: raise ReviewError('样本事实已更新，请重新查看。', 409)
            if previous.get('queryOrigin'):
                from query_review import mask_evidence
                recorded=row.get('assetEvidence') or {}
                actual=mask_evidence(self.source_path,row.get('assets') or {},row['changed_px'])
                if recorded.get('status')!='pixel_count_matches' or actual.get('status')!='pixel_count_matches':
                    raise ReviewError(actual['message'],409)
                if actual.get('maskSHA256')!=recorded.get('maskSHA256'):
                    raise ReviewError('当前掩膜已变化，请重新打开查询结果核对。',409)
            response = {'ok': True, 'sampleId': sid, 'reviewStatus': 'manual_verified',
                        'scope': 'local_demo', 'message': '人工复核结论已保存；后续排除已复核的清单会跳过此样本。未修改原始影像或掩膜。'}
            db.execute('INSERT INTO review_events(owner,request_id,request_hash,source_version,sample_id,facts_hash,note,actor,response,created) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (owner, request_id, request_hash, previous['sourceVersion'], sid, row['facts_hash'], note,
                        str(payload.get('_authenticatedUser') or 'local'), json.dumps(response, ensure_ascii=False), time.time()))
        return response
