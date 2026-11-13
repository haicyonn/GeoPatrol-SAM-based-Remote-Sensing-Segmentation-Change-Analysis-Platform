"""Compact SQL-grounded measure memory for aggregate-query corrections.

The model resolves language; the program checks retained measures. This module
does not author answer SQL, keep answer values, or fine-tune model weights.
"""
import copy
import json
import re
import sqlglot
from sqlglot import exp

LEVELS={'high':'高风险','medium':'中风险','low':'低风险'}
FIELDS={'estimated_building_count':('建筑量','建筑数量','建筑总数','建筑物数量','楼栋数'),
        'foreground_pixels':('建筑像素','建筑像素数'),'foreground_ratio':('建筑占比','建筑覆盖率','建筑像素占比','建筑比例','覆盖比例'),
        'component_count':('连通区域','连通块'),'changed_px':('变化像素','变化面积'),'object_count':('对象数','对象数量','目标数'),
        'changed_ratio':('变化比例',),'hole_ratio':('空洞比例','空洞区域比例','空洞面积比例','hole_ratio'),
        'small_fragment_area_ratio':('碎片面积比例','碎片面积占比','small_fragment_area_ratio')}
CUE=re.compile(r'想要的是|要的是|说的是|指的是|理解错|不是|而是|只看|只要|换成|换到|改成|改为|改用|其余|仍然|沿用|^那|接着|再按|只取|^按|^对比|^比较')
RESET=re.compile(r'新问题|换个问题|另一个问题|重新开始|另外(?:问|查)|单独(?:问|查)')
FLAGS={'has_holes':'有空洞标记','has_fragments':'有碎片标记','border_contact':'有边界接触标记','is_tiny':'有极小变化标记'}
HOLE_CHOICES=('有空洞标记的样本占区域全部样本的百分比','每张样本内部空洞面积比例hole_ratio的均值')


def mentioned_fields(question):
    """Prefer the longest metric phrase at overlapping character positions."""
    matches=[]
    for field,words in FIELDS.items():
        for word in words:
            matches.extend((m.start(),m.end(),field) for m in re.finditer(re.escape(word),question))
    occupied=set();selected=set()
    for start,end,field in sorted(matches,key=lambda x:-(x[1]-x[0])):
        positions=set(range(start,end))
        if not occupied & positions:
            occupied.update(positions);selected.add(field)
    return [field for field in FIELDS if field in selected]


def hole_choice(question):
    if re.fullmatch(r'\s*(?:第?[一1](?:个|种|项)?|前者|(?:我看|先看|就看|看)?样本占比)[。！!\s]*',question) or re.search(r'空洞标记|有空洞的样本',question):return 0
    if re.fullmatch(r'\s*(?:第?[二2](?:个|种|项)?|后者)[。！!\s]*',question) or re.search(r'hole_ratio|面积比例|平均空洞|空洞.{0,8}(?:平均|均值)',question,re.I):return 1
    return None


def ambiguous_hole_metric(question):
    return bool(re.search(r'空洞(?:区域|面积)?(?:的)?(?:比例|占比|率)',question) and hole_choice(question) is None
        and not re.search(r'样本|影像|切片|单张|每张|前\d+|[0-9一二两三四五六七八九十]+张',question))


def explicit_measures(question):
    """Only unambiguous report measures; row selection is a different task."""
    if re.search(r'前\s*[0-9一二两三四五六七八九十]+|[0-9一二两三四五六七八九十]+\s*张|样本ID|sample_id|列出|挑出|筛出',question,re.I):return []
    if re.search(r'平均|均值',question) and not re.search(r'高于|低于|超过|大于|小于',question):
        return [dict(operation='avg',field=field,risk=None) for field in mentioned_fields(question)]
    marker_names=[marker.removeprefix('有').removesuffix('标记') for marker in FLAGS.values()]
    if sum(name in question for name in marker_names)>1:return []
    for field,marker in FLAGS.items():
        name=marker.removeprefix('有').removesuffix('标记')
        if marker.removeprefix('有') in question or re.search(r'(?:有|带)'+re.escape(name)+r'(?:标记)?(?:的)?样本',question):
            if re.search(r'占比|比例|百分',question):return [dict(operation='flag_percentage',field=field,risk=None)]
            if re.search(r'数量|多少|总数',question):return [dict(operation='flag_count',field=field,risk=None)]
    return []


def measure_family(measure):
    from query_clarification import family_for_field
    family=family_for_field(measure['field'])
    if family!=measure['field']:return family
    if measure['field']=='review_level':return 'risk'
    return measure['field']


def metric_change(question,history):
    """An explicitly named new report target replaces the previous projection.

    A marker used only as a WHERE filter ("只看有空洞标记的样本") is not a
    new reported measure. Scope and sample-reference binding remain separate.
    """
    frame=task_frame(history)
    if not frame:return None
    requested=explicit_measures(question)
    if requested:
        families={measure_family(m) for m in requested}
    else:
        hole_target=bool(re.search(r'空洞(?:区域|面积)?(?:的)?(?:比例|占比|率)|hole_ratio',question,re.I))
        from query_semantics import risk_level
        if re.search(r'只看.{0,24}(?:高于|低于|超过|大于|小于)',question):families=set()
        elif hole_target:families={'holes'}
        elif risk_level(question):families={'risk'}
        else:families=set(mentioned_fields(question))
    previous={measure_family(m) for m in frame['measures']}
    if families and (not families<=previous or requested and any(m not in frame['measures'] for m in requested)):
        return dict(previous=frame['measures'],requested=requested,families=sorted(families))
    return None


def _flag(node):
    if node is None:return None
    fields=set()
    for term in node.find_all(exp.EQ):
        a,b=term.this,term.expression
        if isinstance(a,exp.Literal):a,b=b,a
        if isinstance(a,exp.Column) and a.name in FLAGS and isinstance(b,exp.Literal) and str(b.this)=='1':fields.add(a.name)
    return next(iter(fields)) if len(fields)==1 else None

class DialogueError(ValueError):pass

def _risk(node):
    if node is None:return None
    found=set()
    for term in node.find_all(exp.EQ):
        a,b=term.this,term.expression
        if isinstance(a,exp.Literal):a,b=b,a
        if isinstance(a,exp.Column) and a.name=='review_level' and isinstance(b,exp.Literal) and b.this in LEVELS:found.add(b.this)
    return next(iter(found)) if len(found)==1 else None

def report_tree(sql):
    """Inspect aggregates through transparent projection wrappers only.

    This is an analysis view, never replacement execution SQL. A wrapper with
    filters, joins, grouping, ordering or pagination is not transparent.
    """
    tree=sqlglot.parse_one(sql,read='sqlite')
    for _ in range(8):
        if not isinstance(tree,exp.Select) or tree.ctes:break
        source=(tree.args.get('from_') or exp.From()).this
        if not isinstance(source,exp.Subquery) or not isinstance(source.this,exp.Select):break
        if any(tree.args.get(key) for key in ('joins','where','group','having','qualify','order','limit','offset','distinct')):break
        inner=source.this
        names=[p.alias_or_name for p in inner.expressions]
        if not all(names) or len(names)!=len(set(names)):break
        available=dict(zip(names,inner.expressions));projections=[]
        for projection in tree.expressions:
            column=projection.this if isinstance(projection,exp.Alias) else projection
            if not isinstance(column,exp.Column) or column.is_star or column.table and column.table!=source.alias:break
            if column.name not in available:break
            value=available[column.name].copy()
            if isinstance(projection,exp.Alias):
                core=value.this if isinstance(value,exp.Alias) else value
                value=exp.alias_(core,projection.alias,quoted=True)
            projections.append(value)
        else:
            tree=inner.copy();tree.set('expressions',projections)
            continue
        break
    return tree


def frame_from_sql(sql,question):
    """Recover supported aggregate identities from executed SQL, not prose."""
    try:tree=report_tree(sql)
    except (ValueError,sqlglot.errors.ParseError):return None
    if not isinstance(tree,exp.Select) or tree.ctes or tree.find(exp.Subquery):return None
    relations=([tree.args['from_'].this] if tree.args.get('from_') else [])+[j.this for j in tree.args.get('joins') or []]
    if not relations or any(not isinstance(t,exp.Table) or t.name not in {'samples','regions'} for t in relations):return None
    where=tree.args.get('where');where_risk=_risk(where);where_flag=_flag(where)
    measures=[]
    for projection in tree.expressions:
        core=projection.this if isinstance(projection,exp.Alias) else projection
        aggregates=[a for a in core.find_all(exp.AggFunc) if a.find_ancestor(exp.Select) is tree]
        if not aggregates:continue
        flag=_flag(core)
        flag_columns={a.this.name for a in aggregates if isinstance(a,(exp.Sum,exp.Avg)) and isinstance(a.this,exp.Column) and a.this.name in FLAGS}
        if len(flag_columns)==1:flag=next(iter(flag_columns))
        if not flag and where_flag and not _risk(core) and all(isinstance(a,exp.Count) for a in aggregates):flag=where_flag
        if flag:
            measures.append(dict(operation='flag_percentage' if core.find(exp.Div) or isinstance(core,exp.Avg) or core.find(exp.Avg) else 'flag_count',field=flag,risk=None))
            continue
        risk=_risk(core)
        if not risk and where_risk and all(isinstance(a,exp.Count) for a in aggregates):risk=where_risk
        if risk and any(isinstance(a,(exp.Sum,exp.Count,exp.Avg)) for a in aggregates):
            measure=dict(operation='risk_percentage' if core.find(exp.Div) or isinstance(core,exp.Avg) else 'risk_count',field='review_level',risk=risk)
            measures.append(measure);continue
        for aggregate in aggregates:
            if isinstance(aggregate,exp.Count) and isinstance(aggregate.this,(exp.Star,exp.Column)):
                measures.append(dict(operation='count',field='*',risk=None))
            elif isinstance(aggregate,(exp.Avg,exp.Sum,exp.Max,exp.Min)) and isinstance(aggregate.this,exp.Column) and aggregate.this.name in FIELDS:
                measures.append(dict(operation=aggregate.key,field=aggregate.this.name,risk=None))
    unique=[]
    for m in measures:
        if m not in unique:unique.append(m)
    # Supporting output counts are not automatically new user goals.
    if re.search(r'占比|比例|百分',question):
        percentages=[m for m in unique if m['operation'] in {'risk_percentage','flag_percentage'}]
        both=bool(re.search(r'数量.{0,16}(?:和|及|、|以及).{0,12}(?:占比|比例|百分)|(?:占比|比例|百分).{0,16}(?:和|及|、|以及).{0,12}数量',question))
        if percentages:unique=([m for m in unique if m['operation'] in {'risk_count','flag_count'}] if both else [])+percentages
    elif re.search(r'[高中低小]风险',question):
        counts=[m for m in unique if m['operation']=='risk_count']
        if counts:unique=counts
    if not unique:return None
    selection=None
    order=(tree.args.get('order') or exp.Order()).expressions
    limit=tree.args.get('limit')
    if order and limit and limit.expression.sql()=='1' and tree.args.get('group'):
        first=order[0].this
        if isinstance(first,exp.Column) and not first.table:first=next((p.this for p in tree.expressions if isinstance(p,exp.Alias) and p.alias==first.name),first)
        if isinstance(first,exp.Count) and (isinstance(first.this,exp.Star) or isinstance(first.this,exp.Column) and first.this.name=='sample_id'):
            selection=dict(kind='total_sample_count',direction='DESC' if order[0].args.get('desc') else 'ASC')
    return dict(version=1,question=question,measures=unique,regionSelection=selection,
        filters=where.this.sql(dialect='sqlite') if where else None,
        grouping=[g.sql(dialect='sqlite') for g in (tree.args.get('group') or exp.Group()).expressions])

def task_frame(history):
    if not history:return None
    # Re-derive even a stored frame from the successful SQL, so stale/injected
    # prose cannot establish an obligation. History is already version-bound.
    frame=frame_from_sql(history.get('sql',''),(history.get('queryContract') or {}).get('question') or history.get('question',''))
    confirmed=history.get('confirmedMeasures')
    if frame and confirmed and all(m in frame['measures'] for m in confirmed):frame['measures']=copy.deepcopy(confirmed)
    return frame

def label(measure):
    if measure['risk']:return LEVELS[measure['risk']]+('样本百分比' if measure['operation']=='risk_percentage' else '样本数量')
    if measure['field'] in FLAGS:return FLAGS[measure['field']]+('的样本百分比' if measure['operation']=='flag_percentage' else '的样本数量')
    field=(FIELDS.get(measure['field']) or (measure['field'],))[0]
    if measure['operation']=='count':return '全部样本数量'
    return field+{'avg':'均值','sum':'总和','max':'最大值','min':'最小值'}.get(measure['operation'],measure['operation'])

def context_request(question,history):
    if RESET.search(question) or not CUE.search(question):return None
    frame=task_frame(history)
    if not frame:return None
    if metric_change(question,history):return None
    # A request for concrete rows changes report mode, even if it starts with
    # "那" or "只看". The previous region may still be resolved by the model.
    if re.search(r'样本ID|sample_id|[0-9一二两三四五六七八九十]+\s*(?:张|幅|个).{0,8}(?:样本|图片|影像|图像|图)|挑.{0,20}(?:样本|图片|影像)',question,re.I):return None
    from query_semantics import risk_level,regional_total_rank
    expected=copy.deepcopy(frame['measures'])
    new_risk=risk_level(question)
    # Explicitly replacing the measure clears inheritance; a region-selection
    # COUNT in "样本最多的区域" does not replace the reported measure.
    total_only=bool(re.search(r'(?:只(?:要|看|统计|返回)|改(?:看|查)).{0,8}(?:总样本数|总数|样本总数|样本数量)',question) and not regional_total_rank(question))
    if total_only or re.search(r'不(?:看|要|统计).{0,5}风险',question):expected=[]
    if new_risk:
        for m in expected:
            if m['risk']:m['risk']=new_risk
    if re.search(r'占比|比例|百分',question) and '排序指标：' not in question and not re.search(r'只看.{0,24}(?:高于|低于|超过|大于|小于)',question):
        both=bool(re.search(r'数量.{0,10}(?:和|及|、).{0,8}(?:占比|比例|百分)',question))
        if not both:
            for m in expected:
                if m['operation']=='risk_count':m['operation']='risk_percentage'
    # Converting count to percentage must not create duplicate goals when the
    # preceding report already returned both count and percentage.
    unique=[]
    for m in expected:
        if m not in unique:unique.append(m)
    expected=unique
    compare_regions=bool(re.search(r'(?:对比|比较).{0,12}(?:其他|别的|其它)区域|(?:其他|别的|其它)区域.{0,8}(?:相比|比较|对比)',question))
    sort_direction='DESC' if re.search(r'从高到低|从大到小|降序',question) else 'ASC' if re.search(r'从低到高|从小到大|升序',question) else None
    extreme_direction=None
    if not regional_total_rank(question) and re.search(r'只看|只要|只取',question) and re.search(r'区域',question):
        extreme_direction='DESC' if re.search(r'最高|最大|最多',question) else 'ASC' if re.search(r'最低|最小|最少',question) else None
    return dict(frame=frame,required=expected,original=question,compare_regions=compare_regions,sort_direction=sort_direction,extreme_direction=extreme_direction)

def resolution_prompt(request):
    task={k:v for k,v in request['frame'].items() if k!='question'}
    task['previousQuestion']=request['frame']['question'].split('；统计目标：',1)[0][:500]
    return ('你只负责补全数据库对话的当前问题，不写SQL，不回答数值。只输出JSON：'
        '{"mode":"edit或new或clarify","question":"补全后的独立中文问题","clarification":null}。'
        '上一轮任务由已执行SQL提取，不包含答案。用户在纠正/缩小范围时，只修改他明确改动的部分，保留未修改的统计目标、风险等级和口径。'
        '区域的选择指标与选中区域内要报告的统计指标是两个不同角色；改选哪个区域，不等于取消原统计指标。'
        '原来问数量就继续问数量，原来问占比就继续问占比，不能无故互换。明确换统计目标时才能清除旧指标。'
        '“对比其他区域”保留原指标，取消单一区域限制，按各区域分组；没有明确排除时保留原区域作为比较基准。其他非区域筛选条件仍保留。'
        '“按这个比例排序”“只看最高的那个区域”继续使用原指标；后者按该指标排序选区域，不能改按样本总数。'
        '“样本最多/最少的区域”按区域全部样本总数选择，不能按高/低风险数量选择。'
        '编辑时mode=edit；确实另起问题时mode=new；无法消除歧义才mode=clarify并具体提问。'
        'question必须保留本轮选择区域的条件，并明确写出保留的统计目标，而不是只描述区域选择。'
        '\n已执行的任务结构='+json.dumps(task,ensure_ascii=False)
        +'\n本轮未取消、必须保留的指标='+json.dumps([label(m) for m in request['required']],ensure_ascii=False)
        +'\n本轮用户原话='+json.dumps(request['original'],ensure_ascii=False))

def validate_question(question,required):
    from query_semantics import risk_level
    for m in required:
        if m['field'] in FLAGS:
            if FLAGS[m['field']] not in question:raise DialogueError('补全问题丢失应保留的指标：'+label(m))
            if m['operation']=='flag_percentage' and not re.search(r'占比|比例|百分',question):raise DialogueError('必须保留标记样本占比，不能改成单张指标或样本明细。')
        elif m['risk']:
            if risk_level(question)!=m['risk']:raise DialogueError('补全问题丢失或改变了应保留的'+label(m)+'。修改区域选择不能删除原统计目标。')
            percentage=bool(re.search(r'占比|比例|百分',question))
            mixed=any(n['risk']==m['risk'] and n['operation']=='risk_percentage' for n in required) and bool(re.search(r'数量',question))
            if percentage!=(m['operation']=='risk_percentage') and not (m['operation']=='risk_count' and mixed):
                raise DialogueError('必须保留'+label(m)+'的计算方式，不能擅自把数量和占比互换。')
        elif m['field'] in FIELDS:
            if not any(word in question for word in FIELDS[m['field']]):raise DialogueError('补全问题丢失应保留的指标：'+label(m))
            operation={'avg':r'平均|均值','sum':r'合计|总和|总量','max':r'最大','min':r'最小'}.get(m['operation'])
            if operation and not re.search(operation,question):raise DialogueError('补全问题改变了应保留的计算方式：'+label(m))

def parse_resolution(output,request):
    if isinstance(output,dict):
        if output.get('finishReason') in {'length','timeout','cancelled'}:raise DialogueError('上下文补全未完成。')
        text=str(output.get('text') or '')
    else:text=str(output)
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
    try:value=json.loads(text)
    except (ValueError,TypeError) as exc:raise DialogueError('上下文补全必须为JSON。') from exc
    if not isinstance(value,dict) or set(value)!={'mode','question','clarification'} or value['mode'] not in {'edit','new','clarify'}:
        raise DialogueError('上下文补全仅允许mode、question、clarification。')
    if value['mode']=='clarify':
        if not isinstance(value['clarification'],str) or not 1<=len(value['clarification'])<=800:raise DialogueError('需要具体说明上下文缺少什么。')
        return value
    if value['clarification'] is not None or not isinstance(value['question'],str) or not 1<=len(value['question'])<=2000:raise DialogueError('补全问题无效。')
    if request['required'] and value['mode']!='edit':raise DialogueError('本轮是修改原任务，未取消原指标，须mode=edit并保留统计目标。')
    from query_semantics import regional_total_rank,risk_level
    direction=regional_total_rank(request['original'])
    if not direction and risk_level(request['original']) and not re.search(r'区域|全部|所有|各|最高|最低|最多|最少',request['original']):
        previous_selection=request['frame'].get('regionSelection') or {}
        if previous_selection.get('kind')=='total_sample_count':direction=previous_selection['direction']
    # Carry immutable measure slots in program state. Do not ask a small model
    # to remember them again in a prose rewrite. Reject contradictory rewrites,
    # then bind omitted targets visibly instead of inventing replacement SQL.
    model_question=value['question']
    scope_question=('选出样本数量'+('最多' if direction=='DESC' else '最少')+'的一个区域；本轮补充要求：'+request['original']) if direction else model_question
    if not direction and request.get('currentRegions') and not request.get('compare_regions'):
        scope_question=request['original']
    mentioned_risk=risk_level(model_question)
    required_risks={m['risk'] for m in request['required'] if m['risk']}
    if mentioned_risk and required_risks and mentioned_risk not in required_risks:raise DialogueError('本轮未更换风险等级，不能在改写中换成其他风险等级。')
    if request['required'] and all(m['operation']=='risk_count' for m in request['required']) and re.search(r'占比|比例|百分',model_question):
        raise DialogueError('上一轮要数量，本轮只改范围，不能在改写中改成占比。')
    if request['required']:
        value['question']=scope_question.rstrip('。？?；;')+'；统计目标：'+'、'.join(label(m) for m in request['required'])+'。'
        if request.get('compare_regions'):
            value['question']+='本轮按各区域分别比较，上一轮的区域只作对照，不保留单一区域筛选；返回各区域标识和同一指标。'
        if request.get('sort_direction'):
            value['question']+='本轮明确要求按上述统计指标'+('降序' if request['sort_direction']=='DESC' else '升序')+'排列。'
        if request.get('extreme_direction'):
            value['question']+='本轮在上轮候选区域中按上述指标'+('降序' if request['extreme_direction']=='DESC' else '升序')+'选第1个区域，保留所问统计值和区域标识。'
        if any('region_' in g for g in request['frame']['grouping']) and not re.search(r'不要.{0,5}区域|只要数量',request['original']):
            value['question']+='返回区域标识和统计值。'
    validate_question(value['question'],request['required'])
    value['modelQuestion']=model_question
    value['programBoundMeasures']=bool(request['required'])
    value['programBoundSelection']=scope_question!=model_question
    return value

def validate_measures(sql,required):
    actual=frame_from_sql(sql,'')
    if required and not actual:raise DialogueError('当前SQL无法核对继承的统计目标；请直接按区域聚合并保留指标。')
    for measure in required:
        if measure not in actual['measures']:raise DialogueError('SQL丢失或改变上一轮保留的指标：'+label(measure)+'。只改区域/排序时不得改成单纯样本总数。')
        if measure['operation']=='flag_percentage':_validate_flag_percentage(sql,measure['field'])


def validate_measure_order(sql,required,direction):
    tree=sqlglot.parse_one(sql,read='sqlite')
    orders=(tree.args.get('order') or exp.Order()).expressions
    if not orders or bool(orders[0].args.get('desc'))!=(direction=='DESC'):
        raise DialogueError('必须按保留的统计指标'+direction+'排序，不能忽略本轮排序或极值要求。')
    target=orders[0].this
    if isinstance(target,exp.Column) and not target.table:
        target=next((p.this for p in tree.expressions if isinstance(p,exp.Alias) and p.alias==target.name),target)
    probe=tree.copy();probe.set('expressions',[target.copy()]);probe.set('order',None);probe.set('limit',None)
    ordered=frame_from_sql(probe.sql(dialect='sqlite'),'')
    # When count and percentage are both reported, terse "this percentage"
    # and the prior percentage ranking refer to the percentage measure.
    preferred=[m for m in required if m['operation'].endswith('percentage')] or required
    if not ordered or not any(m in ordered['measures'] for m in preferred):
        raise DialogueError('排序必须使用本轮保留的统计指标，不能改成区域总样本数、区域名或单张原始指标。')


def _validate_flag_percentage(sql,field):
    """Bounded proof of conditional sample percentage, not per-image area."""
    tree=report_tree(sql)
    def real_cast(node):
        while isinstance(node,(exp.Alias,exp.Paren)):node=node.this
        return isinstance(node,exp.Cast) and node.args['to'].sql().upper() in {'REAL','FLOAT','DOUBLE'}
    def bare(node):
        while isinstance(node,(exp.Alias,exp.Paren,exp.Round)) or real_cast(node):node=node.this
        return node
    def number(node,value):return isinstance(node,exp.Literal) and not node.is_string and float(node.this)==value
    def own_field(node,name):return isinstance(node,exp.Column) and node.name==name
    def count_all(node):
        node=bare(node)
        if isinstance(node,exp.Nullif) and number(node.expression,0):node=bare(node.this)
        return isinstance(node,exp.Count) and (isinstance(node.this,exp.Star) or own_field(node.this,'sample_id'))
    def conditional_count(node):
        node=bare(node)
        if isinstance(node,exp.Sum) and own_field(node.this,field):return True
        if not isinstance(node,(exp.Sum,exp.Count)) or not isinstance(node.this,exp.Case):return False
        branches=node.this.args.get('ifs') or []
        if len(branches)!=1:return False
        branch=branches[0];condition=branch.this
        if not isinstance(condition,exp.EQ):return False
        a,c=condition.this,condition.expression
        if isinstance(a,exp.Literal):a,c=c,a
        if not own_field(a,field) or not number(c,1) or not number(branch.args.get('true'),1):return False
        default=node.this.args.get('default')
        return number(default,0) if isinstance(node,exp.Sum) else default is None or isinstance(default,exp.Null)
    def numerator(node):
        node=bare(node)
        if not isinstance(node,exp.Mul):return False
        floating_cast=real_cast(node.this) or real_cast(node.expression)
        a,b=bare(node.this),bare(node.expression)
        if number(b,100):a,b=b,a
        return number(a,100) and bool(re.search(r'[.eE]',str(a.this)) or floating_cast) and conditional_count(b)
    where=tree.args.get('where')
    if where and any(col.name==field for col in where.find_all(exp.Column)):
        raise DialogueError('标记样本占比的分母必须保留区域全部样本，不能先WHERE筛掉未标记样本。')
    for projection in tree.expressions:
        core=bare(projection)
        if isinstance(core,exp.Div) and count_all(core.expression) and numerator(core.this):return
        if isinstance(core,exp.Mul):
            a,b=bare(core.this),bare(core.expression)
            if number(b,100):a,b=b,a
            if number(a,100) and isinstance(b,exp.Avg) and own_field(b.this,field):return
            # Division before scaling is valid only when already promoted to
            # REAL; a floating factor after integer division cannot rescue it.
            if number(a,100) and isinstance(b,exp.Div) and count_all(b.expression) and conditional_count(b.this) and (real_cast(b.this) or real_cast(b.expression)):return
    raise DialogueError('标记样本百分比须用100.0乘标记条件计数并除以全部COUNT(*)，或100.0乘布尔标记AVG；不能返回单张hole_ratio、整数截断或使用非空面积字段作分母。')
