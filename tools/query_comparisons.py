"""Typed two-region arithmetic over separately visible SQL operands.

SQL is model-authored and checked against a bounded operand-query shape.
Arithmetic is program-owned; no cached numeric answers enter SQL prompts.
The rank-only LoRA is deliberately not used by this path.
"""
import copy,json,math,re
import sqlglot
from sqlglot import exp

CUE=re.compile(r'相差|差(?:了|多少|几个|几张|值)|(?:少|多|高|低)(?:了)?(?:多少|几|百分)|几倍|多少倍|相比|相较|相对|比起来|对比')
FOLLOW=re.compile(r'^(?:那|再|反过来|倒过来|这两个)|^(?:数量|占比|比例|均值|差值|百分比|百分点|几倍).{0,12}[呢吗？?]?$')
EXCLUDE=re.compile(r'写|解释|为什么|怎么|如何|原理|翻译|不要查|不查数据|新问题|样本ID|sample_id|每张|第[0-9一二三四五六七八九十]+张|排名|第[0-9一二三四五六七八九十]+(?:名|高|低)')


class ComparisonError(ValueError):
    def __init__(self,message,measures=None):super().__init__(message);self.measures=measures or []


def ordered_regions(q,regions):
    found=[]
    for r in regions:
        matches=[m.start() for name in (r['id'],r.get('short',''),r.get('name','')) if name for m in re.finditer(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',q,re.I)]
        if matches:found.append((min(matches),r['id']))
    return [rid for _,rid in sorted(found)]


def is_followup(q,comparison):
    return bool(comparison and FOLLOW.search(q) and not EXCLUDE.search(q) and (CUE.search(q) or re.search(r'反过来|倒过来|数量|占比|比例|均值|百分点|百分之|几倍|多少倍',q)))


def requested(q,history,regions):
    if EXCLUDE.search(q):return False
    if re.search(r'(?:大于|小于|超过|至少|至多|高于|低于)\s*[0-9]|阈值',q):return False
    if re.search(r'平方米|公顷|平方公里|拍摄|采集|真值|准确率|误检率|真实建筑|真实车辆',q):return False
    named=ordered_regions(q,regions)
    return bool(len(named)>=2 and CUE.search(q) or is_followup(q,(history or {}).get('comparisonTask')))


def _measure(q,history):
    from query_dialogue import FIELDS,FLAGS,task_frame
    from query_semantics import risk_level
    frame=task_frame(history) or {};old=(history or {}).get('comparisonTask') or {}
    previous=frame.get('measures') or ([old['measure']] if old else [])
    share=bool(re.search(r'占比|比例|百分比|百分点',q))
    level=risk_level(q)
    if not level and re.search(r'总样本|样本总数|样本总量|全部样本|总数量',q):return dict(operation='count',field='*',risk=None)
    from query_dialogue import mentioned_fields
    for field in mentioned_fields(q):
        words=FIELDS[field]
        operation='avg' if re.search(r'平均|均值',q) else 'sum' if re.search(r'总和|总量|合计|总变化',q) else None
        if operation:return dict(operation=operation,field=field,risk=None)
        matches=[m for m in previous if m['field']==field]
        if len(matches)==1:return matches[0]
        raise ComparisonError('请说明要比较'+words[0]+'的均值还是总和。',[dict(operation=op,field=field,risk=None) for op in ('avg','sum')])
    if level:return dict(operation='risk_percentage' if share else 'risk_count',field='review_level',risk=level)
    for field,marker in FLAGS.items():
        if marker in q or re.search(r'(?:有|带)'+re.escape(marker[1:].removesuffix('标记'))+r'(?:标记)?(?:的)?样本',q):
            return dict(operation='flag_percentage' if share else 'flag_count',field=field,risk=None)
    if previous:
        risks={m['risk'] for m in previous if m.get('risk')}
        if len(risks)==1 and (share or re.search(r'样本|数量|风险量|数目',q)):
            return dict(operation='risk_percentage' if share else 'risk_count',field='review_level',risk=next(iter(risks)))
        if share:
            matches=[m for m in previous if m['operation'].endswith('percentage') or m['operation']=='avg' and m['field'].endswith('_ratio')]
        elif re.search(r'样本|数量|个数|几张|几个|风险量|数目',q):matches=[m for m in previous if m['operation'].endswith('count')]
        else:matches=previous
        if len(matches)==1:return copy.deepcopy(matches[0])
    if re.search(r'样本|数量|几张|几个',q) and not re.search(r'风险|空洞|碎片|占比|比例',q):return dict(operation='count',field='*',risk=None)
    raise ComparisonError('要比较哪一个统计指标？请说明是样本数量、样本占比，还是某个字段的均值。',previous)


def _nonregional_filter(history):
    """Preserve separable AND filters; do not silently rewrite mixed OR scopes."""
    from query_dialogue import task_frame
    if not task_frame(history):return None
    tree=sqlglot.parse_one(history['sql'],read='sqlite')
    if tree.ctes or tree.find(exp.Subquery) or tree.args.get('having'):raise ComparisonError('上一轮包含嵌套统计或分组筛选，请写明这次两个区域需要保留的筛选条件。')
    where=tree.args.get('where')
    if not where:return None
    def terms(n):
        if isinstance(n,exp.Paren):return terms(n.this)
        return terms(n.this)+terms(n.expression) if isinstance(n,exp.And) else [n]
    keep=[]
    for term in terms(where.this):
        columns=list(term.find_all(exp.Column));regional=[c for c in columns if c.name in {'region_id','region_name'}]
        if regional:
            if len(regional)!=len(columns):raise ComparisonError('上一轮的区域条件与其他条件通过OR混合，请明确本次需要保留的筛选条件。')
            continue
        node=term.copy()
        for c in node.find_all(exp.Column):c.set('table',None)
        keep.append(node)
    return exp.and_(*keep).sql(dialect='sqlite') if keep else None


def _replace_filter(filters,field,value):
    tree=sqlglot.parse_one(filters,read='sqlite') if filters else None
    def split(n):return split(n.this)+split(n.expression) if isinstance(n,exp.And) else [n]
    terms=[]
    for term in split(tree) if tree is not None else []:
        if any(c.name==field for c in term.find_all(exp.Column)):
            if isinstance(term,exp.Or):raise ComparisonError('原筛选条件包含OR，请明确本次需要保留的条件。')
        else:terms.append(term)
    if value is not None:terms.append(exp.EQ(this=exp.column(field),expression=exp.Literal.string(value) if isinstance(value,str) else exp.Literal.number(value)))
    return exp.and_(*terms).sql(dialect='sqlite') if terms else None


def plan(q,history,regions,scope,measure_override=None):
    old=(history or {}).get('comparisonTask') or {};named=ordered_regions(q,regions)
    if len(named)>2:raise ComparisonError('这次比较涉及两个区域，请明确哪一个是比较对象、哪一个是基准。')
    if len(named)==2:
        left,right=named
        if re.match(r'\s*(?:与|和|跟|同|相较于|相对于|相比于|以)',q) and re.search(r'相比|相较|相比之下|基准',q):left,right=right,left
    elif old:
        left,right=old['leftRegion'],old['rightRegion']
        if re.search(r'反过来|倒过来',q):left,right=right,left
        elif named:
            if re.search(r'跟|和|与|基准',q):right=named[0]
            else:left=named[0]
    else:raise ComparisonError('请明确要比较的两个区域。')
    if scope and any(r!=scope for r in (left,right)):raise ComparisonError('当前只选择了'+scope.upper()+'，请先切换到全部区域后比较。')
    measure=copy.deepcopy(measure_override) if measure_override else _measure(q,history)
    operation='ratio' if re.search(r'倍',q) else 'relative_change' if re.search(r'百分之|多少\s*[%％]|相对(?:变化|下降|增长)|降幅|增幅',q) else 'difference'
    if re.search(r'反过来|倒过来',q) and old:operation=old['operation']
    filters=old.get('filterExpression') if old and is_followup(q,old) else _nonregional_filter(history)
    from query_semantics import risk_level
    if measure['field']=='review_level':filters=_replace_filter(filters,'review_level',None)
    elif risk_level(q):filters=_replace_filter(filters,'review_level',risk_level(q))
    if re.search(r'未复核|已复核',q):
        if '未复核' in q and '已复核' in q:raise ComparisonError('本次需要保留未复核还是已复核样本？')
        filters=_replace_filter(filters,'reviewed',None if re.search(r'取消|去掉|忽略|不限制',q) else 0 if '未复核' in q else 1)
    return dict(version=1,leftRegion=left,rightRegion=right,measure=measure,operation=operation,filterExpression=filters)


def metric_expression(m):
    from query_dialogue import FIELDS,FLAGS
    operation=m['operation'];field=m['field']
    if operation=='count' and field=='*':return 'COUNT(*)'
    if operation in {'avg','sum','max','min'} and field in FIELDS:return operation.upper()+'('+field+')'
    if operation in {'risk_count','risk_percentage'} and field=='review_level' and m['risk'] in {'high','medium','low'}:
        count="SUM(CASE WHEN review_level = '"+m['risk']+"' THEN 1 ELSE 0 END)"
    elif operation in {'flag_count','flag_percentage'} and field in FLAGS:count='SUM(CASE WHEN '+field+' = 1 THEN 1 ELSE 0 END)'
    else:raise ComparisonError('当前比较不支持这个统计口径。')
    return '100.0 * '+count+' / COUNT(*)' if operation.endswith('percentage') else count


def operand_shape(p):
    # A validation shape and prompt specification, never executable answer SQL.
    expression=metric_expression(p['measure'])
    ids=list(dict.fromkeys((p['leftRegion'],p['rightRegion'])))
    where=exp.In(this=exp.column('region_id'),expressions=[exp.Literal.string(r) for r in ids])
    if p.get('filterExpression'):where=exp.and_(where,sqlglot.parse_one(p['filterExpression'],read='sqlite'))
    return exp.select('region_id',exp.alias_(sqlglot.parse_one(expression,read='sqlite'),'metric_value')).from_('samples').where(where).group_by('region_id')


def validate_operands(sql,p):
    tree=sqlglot.parse_one(sql,read='sqlite');expected=operand_shape(p)
    if not isinstance(tree,exp.Select) or tree.ctes or tree.find(exp.Subquery) or tree.args.get('joins') or tree.args.get('having') or tree.args.get('limit') or tree.args.get('offset') or tree.args.get('distinct') or tree.find(exp.Window):raise ComparisonError('比较必须分别查询两个区域的聚合值；不得用标量差值、连接、子查询或旧分页代替。')
    # Canonicalize benign qualification and AND ordering, without weakening OR.
    def canonical(query):
        t=query.copy();t.set('order',None)
        source=(t.args.get('from_') or exp.From()).this
        if not isinstance(source,exp.Table) or source.name!='samples':raise ComparisonError('操作数必须读取samples，不是regions.sample_count。')
        source.set('alias',None)
        for col in t.find_all(exp.Column):col.set('table',None)
        def conjunction(n):
            if isinstance(n,exp.Paren):return conjunction(n.this)
            return conjunction(n.this)+conjunction(n.expression) if isinstance(n,exp.And) else [n]
        w=t.args.get('where')
        if w:
            pieces=conjunction(w.this)
            for piece in pieces:
                if isinstance(piece,exp.In) and not piece.args.get('query'):piece.set('expressions',sorted(piece.expressions,key=lambda e:e.sql()))
            w.set('this',exp.and_(*sorted(pieces,key=lambda e:e.sql())))
        return t.sql(dialect='sqlite',normalize=True)
    if canonical(tree)!=canonical(expected):raise ComparisonError('两个操作数的区域、指标、分母或筛选条件不符；必须分别按region_id分组并返回region_id、metric_value，计算表达式为'+metric_expression(p['measure'])+'，筛选为'+expected.args['where'].this.sql(dialect='sqlite')+'。')


def summarize(rows,p,regions):
    from query_dialogue import label
    names={r['id']:r.get('short') or r['id'].upper() for r in regions};values={r['region_id']:r['metric_value'] for r in rows}
    count=p['measure']['operation'].endswith('count')
    left=values.get(p['leftRegion'],0 if count else None);right=values.get(p['rightRegion'],0 if count else None)
    if any(v is not None and (not isinstance(v,(float,int)) or not math.isfinite(v)) for v in (left,right)):raise ComparisonError('比较值不是有效数值。')
    title=label(p['measure']);left_name=names[p['leftRegion']];right_name=names[p['rightRegion']]
    scaled=p['measure']['operation']=='avg' and p['measure']['field'].endswith('_ratio')
    share=p['measure']['operation'].endswith('percentage') or scaled
    scale=100 if scaled else 1
    def show(v):return '未知' if v is None else f'{v*scale:.4f}%' if share else f'{v:.6g}'
    detail=f'{left_name}的{title}为{show(left)}，{right_name}为{show(right)}。'
    difference=None if left is None or right is None else left-right
    calculated=difference;unit='个百分点' if share else '个' if count else '原指标单位'
    if difference is None:answer=detail+'存在缺失或没有有效数据，无法计算差值。'
    elif p['operation'] in {'ratio','relative_change'} and right==0:calculated=None;answer=detail+'基准值为0，无法计算相对变化或倍数。'
    elif p['operation']=='ratio':calculated=left/right;unit='倍';answer=detail+f'{left_name}是{right_name}的{calculated:.4f}倍。'
    elif p['operation']=='relative_change':
        calculated=difference/right*100;unit='%';word='减少' if calculated<0 else '增加' if calculated>0 else '变化'
        answer=detail+f'以{right_name}为基准，{left_name}相对{word}{abs(calculated):.4f}%。'
    else:
        calculated=difference*scale;word=('少' if count else '低') if calculated<0 else ('多' if count else '高')
        amount=f'{abs(calculated):.6g}' if count or not share else f'{abs(calculated):.4f}'
        answer=detail+('两者相同，差值为0。' if calculated==0 else f'{left_name}比{right_name}{word}{amount}{unit if unit!="原指标单位" else ""}。')
    return answer,dict(**p,leftValue=left,rightValue=right,signedDifference=difference,value=calculated,unit=unit,label=title,referenceDenominator=right if p['operation'] in {'ratio','relative_change'} else None)


def run_comparison(service,q,scope,data,history,result,invoke,save,finish,clarify,measure_override=None):
    if not requested(q,history,data.source.regions):return None
    try:p=plan(q,history,data.source.regions,scope,measure_override);shape=operand_shape(p)
    except ComparisonError as exc:
        if len(exc.measures)>1:
            from query_clarification import make_pending
            from query_dialogue import label
            options=[]
            for m in exc.measures:
                metric_expression(m)
                replies=['数量','样本数量'] if m['operation'].endswith('count') else ['占比','百分比'] if m['operation'].endswith('percentage') else ['均值','平均'] if m['operation']=='avg' else ['总和','合计']
                options.append(dict(label=label(m),value=dict(type='comparison_measure',measure=m,label=label(m),replies=replies)))
            pending=make_pending(history,q,str(exc),'comparison_metric',options,data.data_version,data.source.regions)
            save({**(history or {}),'pending':pending})
        return clarify(str(exc),'comparison_definition')
    from model_sql import parse_output,read_sql,review_version,Snapshot,StaleData
    SCHEMA=service.schema
    from query_contract import remember
    from query_dialogue import label
    c=dict(action='query',question=q+'；分别统计两个区域的'+label(p['measure'])+'，再计算比较结果。',regions=list(dict.fromkeys((p['leftRegion'],p['rightRegion']))),columns=['region_id','metric_value'],order=[],limit=None,offset=0,per_group=False,calculations=[p['operation']+'，操作数分别来自两个区域，基准为'+p['rightRegion']],missing=None,orderRequired=False)
    result['queryContract']=c;result['comparisonPlan']=p
    task=dict(regions=c['regions'],metricExpression=metric_expression(p['measure']),filter=shape.args['where'].this.sql(dialect='sqlite'),columns=c['columns'])
    sql_prompt=('根据已确认比较任务生成一条只读SQLite SELECT，只返回JSON {"sql":"完整SELECT","clarification":null}。'
        '先分别查两个区域的聚合值，一行一个区域。程序再计算差值，SQL不要计算差值，也不要比较样本总数冒充风险样本数。'
        'FROM samples，GROUP BY region_id；输出region_id和metric_value；必须原样使用metricExpression并AS metric_value，必须原样使用filter。'
        '不要JOIN regions，不要子查询，不要LIMIT/OFFSET。风险计数来自review_level条件聚合；不存在缺少计算别名的问题。'
        '\n'+SCHEMA+'\nCONFIRMED_OPERANDS='+json.dumps(task,ensure_ascii=False))
    error='';sql=''
    for _ in range(2):
        try:
            output=invoke(sql_prompt+('\n修复上次SQL='+sql+'\n错误='+error if error else ''),'sql',950)
            generated,missing=parse_output(output)
            if missing:raise ComparisonError('字段与口径均已给出，不能要求用户提供样本数量或计算别名：'+missing)
            sql=read_sql(generated);validate_operands(sql,p);executed=data.execute(sql,region_id=scope)
            if review_version(service.reviewed_provider())!=data.review_version or Snapshot(service.source_path).source_version!=data.version:raise StaleData('数据版本已变化，请重新比较。')
            answer,comparison=summarize(executed['rows'],p,data.source.regions)
            next_state=remember(history,q,c,executed);next_state['confirmedMeasures']=[p['measure']];next_state['comparisonTask']=p
            save(next_state)
            result['attempts'].append(dict(stage='sql',status='executed',sql=executed['sql'],modelGeneratedSQL=generated))
            result.update(status='completed',grounded=True,answer=answer,comparison=comparison,sqlQuery=executed['sql'],sqlRows=executed['rows'],sqlColumns=executed['columns'],sqlTruncated=executed['truncated'],modelGeneratedSQL=generated,programChanges=[],semanticChecks=['independent_region_operands','bound_comparison_measure','preserved_comparison_filters','deterministic_comparison_arithmetic'])
            return finish()
        except Exception as exc:
            error=str(exc)[:800];result['attempts'].append(dict(stage='sql',status='rejected',sql=sql,error=error))
            if isinstance(exc,StaleData):break
    result.update(status='failed',answer='未能取得符合口径的两个独立统计值，未返回比较结果。',error=error);return finish()
