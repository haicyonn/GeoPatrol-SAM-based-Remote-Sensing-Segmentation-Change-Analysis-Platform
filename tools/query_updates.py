"""Version 1: typed rank edits over an executed regional aggregate task.

The interpreter predicts a delta. The program owns scope, metric and filters;
SQL remains model-authored. Unsupported edits use the existing query planner.
Training and inference share the same prompt, schema and validation functions.
"""
import copy,hashlib,json,re
import sqlglot
from sqlglot import exp

NUMBER=r'[0-9一二两三四五六七八九十百]+'
RANK_CUE=re.compile(r'排名|排第|第\s*'+NUMBER+r'\s*(?:名|位|高|低|大|小|多|少)|最高|最低|最大|最小|最多|最少|榜首|冠军|亚军|季军|倒数|下一[名个位]|上一[名个位]|往[前后].{0,3}一位|前\s*'+NUMBER+r'\s*(?:名|个|位)|从高到低|从低到高|降序|升序')
EXCLUDE=re.compile(r'解释|为什么|怎么|如何|原理|写|翻译|不要查|不查数据|什么是|新问题|换个问题|样本ID|sample_id|每张|\d+张|第'+NUMBER+r'张|挑.{0,12}样本')


class UpdateError(ValueError):pass


def number(text):
    if text.isdigit():return int(text)
    values={'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    if '百' in text:
        a,b=text.split('百',1);return values.get(a,1)*100+(number(b.lstrip('零')) if b else 0)
    if '十' in text:
        a,b=text.split('十',1);return values.get(a,1)*10+values.get(b,0)
    return values.get(text,0)


def literal_rank(question,default_direction='DESC'):
    """Independent numeric constraints, not a second model interpretation."""
    q=question.split('；',1)[0]
    # A tie-break clause describes a second key, not the primary ranking.
    # Preserve semantic extrema ("最低") when only the tie-break has an
    # explicit direction ("均值相同按region_id降序").
    primary_q=re.split(r'[，,;。](?=[^，,;。]*(?:相同|相等|同值|并列|同分|一样))',q,maxsplit=1)[0]
    primary_q=re.split(r'同值|并列时|并列则|相同时|相等时',primary_q,maxsplit=1)[0]
    explicit_order=re.search(r'从低到高|从高到低|升序|降序',primary_q)
    if explicit_order:
        # Later directions may be stable tie-breaks on a different field.
        # The first explicit ORDER BY instruction controls the primary rank.
        direction='ASC' if explicit_order[0] in {'从低到高','升序'} else 'DESC'
    else:
        direction='ASC' if re.search(r'倒数|最低|最小|最少|第'+NUMBER+r'(?:低|小|少)',primary_q) else 'DESC' if re.search(r'最高|最大|最多|冠军|亚军|季军|榜首|第'+NUMBER+r'(?:高|大|多)',primary_q) else default_direction
    interval=re.search(r'第\s*('+NUMBER+r')\s*(?:到|至|—|-)\s*('+NUMBER+r')',q)
    ordinal=re.search(r'(?:第|排名\s*第?|排在第?)\s*('+NUMBER+r')\s*(?:名|位|高|低|大|小|多|少|(?=[的呢？?，,。!！\s]|$))',q)
    top=re.search(r'前\s*('+NUMBER+r')\s*(?:名|个|位)',q)
    if interval:start,end=number(interval[1]),number(interval[2])
    elif ordinal:start=end=number(ordinal[1])
    elif top:start,end=1,number(top[1])
    elif re.search(r'亚军',q):start=end=2
    elif re.search(r'季军',q):start=end=3
    elif re.search(r'最高|最低|最大|最小|最多|最少|榜首|冠军',q):start=end=1
    elif re.search(r'从高到低|从低到高|升序|降序',q):start,end=1,None
    else:return None
    if start<1 or end is not None and (end<start or end-start>=50):raise UpdateError('排名区间必须有效且一次最多50名。')
    return dict(start=start,end=end,direction=direction)


def report_task(history):
    from query_dialogue import frame_from_sql,task_frame
    if not history or not history.get('sql'):return None
    tree=sqlglot.parse_one(history['sql'],read='sqlite');frame=task_frame(history)
    if not frame or tree.ctes or tree.find(exp.Subquery) or tree.find(exp.Window):return None
    columns=[p.alias_or_name for p in tree.expressions]
    if 'sample_id' in columns or not any(c in columns for c in ('region_id','region_name')):return None
    groups=(tree.args.get('group') or exp.Group()).expressions
    if not groups or any(not isinstance(g,exp.Column) or g.name not in {'region_id','region_name'} for g in groups):return None
    projections={p.alias_or_name:p for p in tree.expressions}
    orders=(tree.args.get('order') or exp.Order()).expressions
    metric=None
    if orders:
        target=orders[0].this
        for name,p in projections.items():
            core=p.this if isinstance(p,exp.Alias) else p
            if core.find(exp.AggFunc) and (isinstance(target,exp.Column) and target.name==name or target.sql()==core.sql()):metric=name;break
    aggregate_aliases=[name for name,p in projections.items() if p.find(exp.AggFunc)]
    if metric is None and len(frame['measures'])==1 and len(aggregate_aliases)==1:metric=aggregate_aliases[0]
    if metric is None:return None
    offset=int(tree.args['offset'].expression.this) if tree.args.get('offset') else 0
    limit=int(tree.args['limit'].expression.this) if tree.args.get('limit') else None
    identity='region_id' if 'region_id' in columns else 'region_name'
    probe=tree.copy();probe.set('expressions',[projections[metric].copy()]);metric_frame=frame_from_sql(probe.sql(),'')
    return dict(version=1,kind='regional_aggregate',baseSQL=history['sql'],columns=columns,rankMeasure=metric_frame['measures'][0],
        question=(history.get('queryContract') or {}).get('question') or history.get('question',''),
        candidateRegions=list((history.get('queryContract') or {}).get('regions') or []),
        measures=frame['measures'],filters=frame['filters'],metricAlias=metric,identity=identity,
        rank=dict(start=offset+1,end=offset+limit if limit else None,direction='ASC' if orders and not orders[0].args.get('desc') else 'DESC'),
        id=hashlib.sha256(history['sql'].encode()).hexdigest()[:16])


def eligible(question,task,regions):
    if not task or EXCLUDE.search(question):return False
    # An explicit restart requests full planning even if it repeats the old
    # metric and includes a rank. Do not force it through a rank-only adapter.
    if re.search(r'(?:^|[，,。；;])\s*(?:请(?:帮我)?|帮我|现在|那么|那)?\s*(?:重新(?:查询|统计|分析)|重查|从头(?:查询|统计))',question):return False
    from query_semantics import regional_total_rank
    if regional_total_rank(question) or re.search(r'这个区域|该区域|本区域',question):return False
    try:stated=literal_rank(question,task['rank']['direction'])
    except UpdateError:return True
    if stated is None and not RANK_CUE.search(question):return False
    if stated and stated['end'] is None:return False
    from query_clarification import region_ids
    from query_dialogue import metric_change
    if metric_change(question,dict(sql=task['baseSQL'],queryContract=dict(question=task['question']))):return False
    named=region_ids(question,regions)
    # A named region in a rank assertion is the verification target, never a
    # pre-ranking WHERE condition. Other region edits use the original planner.
    if named and not re.search(r'不是|是不是|是否|对吗|排第|排名|排在',question):return False
    if re.search(r'并列|密集排名|跳名次',question):return False
    return True


def task_input(task):
    return {k:copy.deepcopy(task[k]) for k in ('kind','candidateRegions','measures','filters','rank')}|dict(rankMeasure=copy.deepcopy(task.get('rankMeasure',task['measures'][0])))


def prompt(question,task,regions):
    return ('你是多轮数据查询的状态更新器。只输出JSON，不写SQL，不回答数值。'
        '格式：{"action":"update_query或verify_rank或new_query或chat或clarify","changes":{"rank":{"start":正整数,"end":正整数或null,"direction":"DESC或ASC"}},"target_region":null,"clarification":null}。'
        'update_query只修改名次，统计指标、候选区域、筛选条件保持不变。第N名=start=N,end=N；前N名=start=1,end=N；第a到b名=start=a,end=b。'
        '最高/第一/榜首=start=1,end=1,DESC；最低/倒数第一=ASC；升降序查看全部时start=1,end=null。下一名按已有末名加1。'
        '“UAV不是第二高吗”是verify_rank，target_region="uav"，rank写声称的名次；核查目标不能缩小候选区域。'
        'new_query/chat/clarify时changes={}；仅clarify填写clarification，其余为null。普通闲聊为chat；新指标或新筛选为new_query；缺少原任务才澄清。'
        '第N名未说明高低时沿用TASK.rank.direction；“第二高”DESC，“第二低”ASC。本轮必须保留原任务的统计口径，即使用户省略指标和区域。'
        '\nTASK='+json.dumps(task_input(task),ensure_ascii=False,separators=(',',':'))+
        '\nREGIONS='+json.dumps([dict(id=r['id'],label=r.get('short') or r['id']) for r in regions],ensure_ascii=False,separators=(',',':'))+
        '\nMESSAGE='+json.dumps(question,ensure_ascii=False))


def parse_update(output,question,task,regions):
    if isinstance(output,dict):
        if output.get('finishReason') not in (None,'stop'):raise UpdateError('任务更新没有完整生成。')
        output=output.get('text','')
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',str(output).strip(),flags=re.I)
    try:value=json.loads(text)
    except ValueError as e:raise UpdateError('任务更新须为JSON。') from e
    if not isinstance(value,dict) or set(value)!={'action','changes','target_region','clarification'}:raise UpdateError('任务更新字段不符合协议。')
    action=value['action']
    if action not in {'update_query','verify_rank','new_query','chat','clarify'}:raise UpdateError('未知任务动作。')
    if action not in {'update_query','verify_rank'}:
        if value['changes']!={} or value['target_region'] is not None:raise UpdateError('非更新动作不得修改任务。')
        if action=='clarify':
            if not isinstance(value['clarification'],str) or not 1<=len(value['clarification'])<=500:raise UpdateError('须说明缺少的条件。')
        elif value['clarification'] is not None:raise UpdateError('仅澄清动作可以提问。')
        if eligible(question,task,regions) and literal_rank(question,task['rank']['direction']):raise UpdateError('已有明确排名任务；本轮仅改名次，不应转聊天、另起任务或重复澄清。')
        return value
    if value['clarification'] is not None or not isinstance(value['changes'],dict) or set(value['changes'])!={'rank'}:raise UpdateError('排名更新只能包含rank。')
    rank=value['changes']['rank']
    if not isinstance(rank,dict) or set(rank)!={'start','end','direction'}:raise UpdateError('rank须包含start/end/direction。')
    if type(rank['start']) is not int or not 1<=rank['start']<=10000 or rank['direction'] not in {'ASC','DESC'}:raise UpdateError('名次或方向无效。')
    if rank['end'] is None:
        if rank['start']!=1:raise UpdateError('完整排序应从第1名开始。')
    elif type(rank['end']) is not int or not rank['start']<=rank['end']<rank['start']+50:raise UpdateError('排名区间无效。')
    explicit=literal_rank(question,task['rank']['direction'])
    if re.search(r'下一[名个位]|往后.{0,3}一位',question) and task['rank']['end'] is not None:
        n=task['rank']['end']+1;explicit=dict(start=n,end=n,direction=task['rank']['direction'])
    if re.search(r'上一[名个位]|往前.{0,3}一位',question) and task['rank']['start']>1:
        n=task['rank']['start']-1;explicit=dict(start=n,end=n,direction=task['rank']['direction'])
    if explicit and rank!=explicit:raise UpdateError('必须符合用户明确给出的排名：'+json.dumps(explicit,ensure_ascii=False))
    from query_clarification import region_ids
    named=region_ids(question,regions)
    assertion=bool(named and re.search(r'不是|是不是|是否|对吗|排第|排名|排在',question))
    if assertion and action!='verify_rank':raise UpdateError('用户在核查指定区域的排名，须verify_rank；不能替换候选范围。')
    if action=='verify_rank':
        if len(named)!=1 or value['target_region']!=named[0]:raise UpdateError('核查目标须与用户明确提及的唯一区域一致。')
    elif value['target_region'] is not None:raise UpdateError('普通排名更新没有核查目标。')
    return value


def contract_for_update(question,task,delta):
    rank=delta['changes']['rank'];verify=delta['action']=='verify_rank'
    return dict(action='query',question=question,regions=task['candidateRegions'],columns=task['columns'],
        order=[dict(expression=task['metricAlias'],direction=rank['direction']),dict(expression=task['identity'],direction='ASC')],
        limit=None if verify or rank['end'] is None else rank['end']-rank['start']+1,
        offset=0 if verify or rank['end'] is None else rank['start']-1,per_group=False,
        calculations=['沿用已执行任务的指标与筛选条件，仅修改排名。'],missing=None,orderRequired=True)


def validate_update_sql(sql,task,contract,scope):
    from query_contract import validate_sql
    validate_sql(sql,contract,scope)
    def body(text):
        tree=sqlglot.parse_one(text,read='sqlite')
        for key in ('order','limit','offset'):tree.set(key,None)
        return tree.sql(dialect='sqlite',normalize=True)
    if body(sql)!=body(task['baseSQL']):raise UpdateError('本轮只改排名，SQL不得改变原任务的统计指标、候选范围、分组、输出列或筛选条件。请复制原SQL主体，只修改ORDER BY和LIMIT/OFFSET。')


def validate_rank_contract(question,contract):
    if contract['action']!='query' or 'sample_id' in contract['columns'] or not any(x in contract['columns'] for x in ('region_id','region_name')) or contract['per_group']:return
    rank=literal_rank(question)
    if rank is None:return
    limit=None if rank['end'] is None else rank['end']-rank['start']+1
    if (contract['limit'],contract['offset'])!=(limit,rank['start']-1):raise UpdateError('原问题要求的名次不符：limit应为'+str(limit)+'，offset应为'+str(rank['start']-1)+'。第N名只取1条，不是第N到N+2条。')
    if not contract['order'] or contract['order'][0]['direction']!=rank['direction']:raise UpdateError('排名方向必须为'+rank['direction'])
    contract['orderRequired']=True


def result_summary(rows,task,delta,regions):
    from query_dialogue import label
    rank=delta['changes']['rank'];identity=task['identity'];metric=task['metricAlias']
    names={r['id']:r.get('short') or r['id'].upper() for r in regions}
    aliases={str(x).lower():r['id'] for r in regions for x in (r['id'],r.get('name',''),r.get('short','')) if x}
    def region(row):return aliases.get(str(row.get(identity,'')).lower(),str(row.get(identity,'')))
    def display(row):
        value=row.get(metric);measure=task['rankMeasure']
        title=label(measure)
        if isinstance(value,(float,int)):
            if measure['field'].endswith('_ratio') and measure['operation']=='avg':return title+f'约{value*100:.4f}%'
            if measure['operation'].endswith('percentage'):return title+f'约{value:.4f}%'
            return title+'为'+f'{value:.6g}'
        return title+'为未知值'
    if delta['action']=='verify_rank':
        found=next(((i+1,row) for i,row in enumerate(rows) if region(row)==delta['target_region']),None)
        target=names[delta['target_region']]
        if not found:return target+'不在当前候选范围或没有对应结果，无法确认所问名次。',dict(targetRegion=delta['target_region'],actualRank=None,claimedRank=rank['start'])
        actual,row=found
        return f'核查后，{target}在当前候选范围按该指标'+('从高到低' if rank['direction']=='DESC' else '从低到高')+f'排第{actual}名，'+display(row)+f'。你提到的是第{rank["start"]}名。',dict(targetRegion=delta['target_region'],actualRank=actual,claimedRank=rank['start'],matchesClaim=actual==rank['start'])
    if not rows:return '当前候选范围没有所问名次，未扩大区域范围或更换统计指标。',None
    if rank['start']==rank['end']:
        row=rows[0];return f'按当前口径，第{rank["start"]}名是{names.get(region(row),region(row))}，'+display(row)+'。',None
    return '已按原指标和候选范围返回'+('完整排序' if rank['end'] is None else f'第{rank["start"]}—{rank["end"]}名')+'；同值时按区域标识升序区分行次。',None


def run_segmentation_measure_switch(service,question,scope,data,history,result,save,finish):
    """A bare metric change keeps a verified scalar SUM's source and filters."""
    if service.task_type!='building_segmentation' or not history or history.get('pending') or not history.get('sql'):return None
    match=re.fullmatch(r'(?:那|那么|改看|换成|改成)?(建筑量|建筑数量|(?:原始)?连通区域(?:总数|数)?)(?:呢|是多少)?',question.strip(' 。？?!！'))
    if not match:return None
    from model_sql import read_sql,review_version,Snapshot,StaleData
    from query_contract import remember
    from query_state import derive_population
    tree=sqlglot.parse_one(history['sql'],read='sqlite');source=(tree.args.get('from_') or exp.From()).this
    if (not isinstance(tree,exp.Select) or not isinstance(source,exp.Table) or source.name!='samples' or len(tree.expressions)!=1
        or tree.ctes or tree.find(exp.Subquery) or tree.find(exp.Window) or any(tree.args.get(k) for k in ('joins','group','having','distinct','limit','offset','order'))):return None
    core=tree.expressions[0];core=core.this if isinstance(core,exp.Alias) else core
    if not isinstance(core,exp.Sum) or not isinstance(core.this,exp.Column) or core.this.name not in {'estimated_building_count','component_count'}:return None
    field='estimated_building_count' if match[1] in {'建筑量','建筑数量'} else 'component_count'
    tree=tree.copy();tree.set('expressions',[exp.alias_(exp.Sum(this=exp.column(field,table=source.alias_or_name)),field)])
    executed=data.execute(read_sql(tree.sql(dialect='sqlite')),region_id=scope)
    if review_version(service.reviewed_provider())!=data.review_version or Snapshot(service.source_path).source_version!=data.version:raise StaleData('数据版本已变化，请重新查询。')
    old=history.get('queryContract') or {}
    c={k:copy.deepcopy(v) for k,v in old.items() if k in {'regions','calculations'}}
    prior_question=(old.get('question') or history.get('question') or '').strip(' 。？?!！')
    rewritten=re.sub(r'(?:原始)?连通区域(?:总数|数)?|建筑量|建筑数量',match[1],prior_question)
    c.update(action='query',question=rewritten+'。',columns=[field],order=[],limit=None,offset=0,per_group=False,missing=None)
    save(remember(history,question,c,executed,derive_population(executed['sql'],executed,question)))
    result['attempts'].append(dict(stage='state_update',status='executed',source='verified_sql_measure_edit',sql=executed['sql']))
    result.update(status='completed',grounded=True,queryContract=c,sqlPlanSource='verified_model_sql_measure_edit',sqlQuery=executed['sql'],sqlRows=executed['rows'],sqlColumns=executed['columns'],sqlTruncated=executed['truncated'],programChanges=['metric_projection'],semanticChecks=['same_sum_source_filters'],answer='沿用上一轮筛选与统计范围，已替换统计指标。')
    return finish()


def run_sample_rank(service,question,scope,data,history,result,save,finish):
    """A literal paging edit reuses verified model SQL; never invent new filters.

    Restricted to a short rank-only message and one direct sample SELECT.
    Queries changing a metric, population or business condition use the model.
    """
    q=question.strip(' 。？?!！')
    match=re.fullmatch(r'(?:那|那么|再|请|看看|看|只看|只取)?(?:排名)?第?('+NUMBER+r')(?:高|低|大|小|多|少|名|位)(?:的)?(?:呢|吧)?',q)
    if not match or not history or history.get('pending') or not history.get('sql'):return None
    from model_sql import read_sql,review_version,Snapshot,StaleData
    from query_contract import remember
    from query_state import derive_population
    tree=sqlglot.parse_one(history['sql'],read='sqlite')
    source=(tree.args.get('from_') or exp.From()).this
    group=tree.args.get('group')
    per_sample_group=bool(group and isinstance(source,exp.Table) and any(isinstance(g,exp.Column) and g.name=='sample_id' and g.table in ('',source.alias_or_name) for g in group.expressions))
    if (not isinstance(tree,exp.Select) or not isinstance(source,exp.Table) or source.name!='samples'
        or tree.ctes or tree.find(exp.Subquery) or tree.find(exp.Window)
        or tree.find(exp.AggFunc) and not per_sample_group or group and not per_sample_group
        or any(tree.args.get(k) for k in ('joins','having','distinct'))
        or 'sample_id' not in [p.alias_or_name for p in tree.expressions] or not tree.args.get('order')):return None
    orders=tree.args['order'].expressions
    if not orders or not isinstance(orders[0].this,exp.Column):return None
    rank=literal_rank(q,'DESC' if orders[0].args.get('desc') else 'ASC')
    if not rank or rank['end']!=rank['start']:return None
    # The wording may change direction; all other SQL nodes remain untouched.
    tree=tree.copy();tree.args['order'].expressions[0].set('desc',rank['direction']=='DESC')
    tree=tree.limit(1).offset(rank['start']-1)
    sql=read_sql(tree.sql(dialect='sqlite'))
    executed=data.execute(sql,region_id=scope)
    if review_version(service.reviewed_provider())!=data.review_version or Snapshot(service.source_path).source_version!=data.version:
        raise StaleData('数据版本已变化，请重新查询。')
    old=history.get('queryContract') or {}
    c={k:copy.deepcopy(v) for k,v in old.items() if k in {'action','regions','columns','calculations','missing'}}
    c.update(action='query',question=history.get('question','')+'；本轮仅取第'+str(rank['start'])+'名。',
             columns=executed['columns'],order=[dict(expression=o.this.sql(dialect='sqlite'),direction='DESC' if o.args.get('desc') else 'ASC') for o in tree.args['order'].expressions],
             limit=1,offset=rank['start']-1,per_group=False,missing=None)
    population=derive_population(sql,executed,question)
    save(remember(history,question,c,executed,population))
    result['attempts'].append(dict(stage='state_update',status='executed',source='verified_sql_rank_edit',sql=sql,rank=rank))
    result.update(status='completed',grounded=True,queryContract=c,sqlPlanSource='verified_model_sql_rank_edit',
        sqlQuery=sql,sqlRows=executed['rows'],sqlColumns=executed['columns'],sqlTruncated=executed['truncated'],
        programChanges=['rank_pagination'],semanticChecks=['same_projection_source_filters','literal_rank_pagination'],
        answer='沿用已校验查询的指标和筛选，仅更新排名分页，返回'+str(len(executed['rows']))+'行。')
    return finish()


def run_update(service,question,scope,data,history,result,invoke,save,finish):
    """Return None for unsupported edits, otherwise a finished normal response."""
    from model_sql import parse_output,read_sql,Snapshot,StaleData,review_version
    from query_contract import remember
    task=report_task(history)
    if (history or {}).get('pending') or not eligible(question,task,data.source.regions):return None
    try:literal_rank(question,task['rank']['direction'])
    except UpdateError as exc:
        from query_clarification import problem
        result.update(status='needs_clarification',answer=str(exc),clarificationSource='invalid_rank',clarification=problem(str(exc),'invalid_rank'))
        return finish()
    delta=None;error='';base_prompt=prompt(question,task,data.source.regions)
    for _ in range(2):
        try:
            output=invoke(base_prompt+('\n修正协议错误：'+error if error else ''),'state_update',300)
            delta=parse_update(output,question,task,data.source.regions)
            result['attempts'].append(dict(stage='state_update',status='resolved',delta=delta));break
        except Exception as exc:
            error=str(exc)[:500];result['attempts'].append(dict(stage='state_update',status='rejected',error=error))
    if delta is None:
        result.update(status='failed',answer='未能可靠更新本次排名，未查询数据。',error=error);return finish()
    if delta['action'] not in {'update_query','verify_rank'}:return None
    c=contract_for_update(question,task,delta);result['queryContract']=c
    result['taskUpdate']=dict(baseTaskId=task['id'],source='model_delta',**delta)
    source=sqlglot.parse_one(task['baseSQL'],read='sqlite')
    for key in ('order','limit','offset'):source.set(key,None)
    sql_prompt=('你是SQLite查询编辑器。只输出JSON {"sql":"完整SELECT","clarification":null}。'
        '原任务已执行且统计口径已确认。基础聚合SQL必须保持原样，仅按契约添加ORDER BY、LIMIT和OFFSET。'
        '所有原输出别名、表、JOIN、WHERE、GROUP BY、HAVING均保持。不要重解释业务，不要询问已有条件。'
        '核查指定区域排名时，先对全部候选区域排序；不能WHERE核查目标。'
        'limit=null表示完整排序，不写LIMIT也不写OFFSET；有limit时严格使用该整数，只取一个名次时LIMIT必须1。'
        '\nBASE_AGGREGATE_SQL='+source.sql(dialect='sqlite')+'\nCONFIRMED_CONTRACT='+json.dumps(c,ensure_ascii=False))
    if delta['action']=='verify_rank':sql_prompt+='\n本轮是排名核查：返回全部候选区域，一行一个区域；禁止LIMIT和OFFSET，禁止按核查目标缩小WHERE。'
    sql='';error=''
    for _ in range(2):
        try:
            output=invoke(sql_prompt+('\n上次SQL='+sql+'\n修正错误：'+error if error else ''),'sql',1100)
            generated,clarification=parse_output(output)
            if clarification:raise UpdateError('条件已确认，请编辑原SQL；计算别名不是缺少字段。')
            sql=read_sql(generated);validate_update_sql(sql,task,c,scope)
            executed=data.execute(sql,region_id=scope)
            if review_version(service.reviewed_provider())!=data.review_version or Snapshot(service.source_path).source_version!=data.version:raise StaleData('数据版本已变化，请重新查询。')
            if executed['truncated']:raise UpdateError('区域排名结果被截断，不能据此确认完整名次。')
            answer,verification=result_summary(executed['rows'],task,delta,data.source.regions)
            next_state=remember(history,question,c,executed)
            next_state['confirmedMeasures']=copy.deepcopy(task['measures'])
            save(next_state)
            result['attempts'].append(dict(stage='sql',status='executed',sql=executed['sql'],modelGeneratedSQL=generated))
            result.update(status='completed',grounded=True,answer=answer,sqlQuery=executed['sql'],sqlRows=executed['rows'],sqlColumns=executed['columns'],sqlTruncated=False,modelGeneratedSQL=generated,programChanges=[],semanticChecks=['immutable_aggregate_body','bound_rank_delta','stable_region_tie_break'])
            if verification:result['rankVerification']=verification
            return finish()
        except Exception as exc:
            error=str(exc)[:500];result['attempts'].append(dict(stage='sql',status='rejected',sql=sql,error=error))
            if isinstance(exc,StaleData):break
    result.update(status='failed',answer='SQL未满足已确认的排名要求，未返回结果。',error=error);return finish()
