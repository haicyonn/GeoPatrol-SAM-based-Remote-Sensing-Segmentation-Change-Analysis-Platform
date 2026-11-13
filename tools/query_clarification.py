"""Version-bound query decisions and typed clarification choices.

This module owns questions and slot updates, never answer SQL or numeric answers.
The metric catalog describes business meanings rather than whole user sentences.
"""
import copy
import json
import re
import uuid

import sqlglot
from sqlglot import exp

CATALOG={
    'holes':dict(names=['空洞','孔洞'],flag='has_holes',field='hole_ratio',label='空洞',marker='有空洞标记',area='内部空洞面积比例'),
    'fragments':dict(names=['碎片'],flag='has_fragments',field='small_fragment_area_ratio',label='碎片',marker='有碎片标记',area='小碎片面积占比'),
}


def family_for_field(field):
    return next((key for key,item in CATALOG.items() if field in {item['flag'],item['field']}),field)


def region_ids(question,regions):
    return [r['id'] for r in regions if any(a and re.search(r'(?<![A-Za-z0-9_])'+re.escape(a)+r'(?![A-Za-z0-9_])',question,re.I)
        for a in (r['id'],r.get('short',''),r.get('name','')))]


def unresolved_region_reference(question,regions):
    match=re.search(r'这个区域|该区域|这个地区|该地区|刚才那个区域',question)
    # “LEVIR中高于该区域均值” has a local antecedent; “该区域和WHU”
    # still needs a distinct antecedent from prior executed results.
    if match and (not region_ids(question,regions) or re.search(r'和|与|对比|相比|比较',question)):return match
    return None


def is_catalog_query(question,regions):
    """Known data scope plus a named statistic is already a data request."""
    if not region_ids(question,regions):return False
    metric=any(name in question for item in CATALOG.values() for name in item['names']) or bool(re.search(r'风险|样本|变化像素|变化面积|对象数|对象数量|changed_px|hole_ratio|review_level',question,re.I))
    statistic=bool(re.search(r'比例|占比|百分|数量|总数|均值|平均|风险等级',question))
    return metric and statistic


def semantic_state(history,regions):
    """Recover the task from executed SQL; never trust a client's task summary."""
    from query_dialogue import task_frame,label,report_tree
    history=history or {};sql=history.get('sql','')
    if not sql:return None
    frame=task_frame(history)
    try:tree=report_tree(sql)
    except (ValueError,sqlglot.errors.ParseError):return None
    where=tree.args.get('where');c=history.get('queryContract') or {}
    measures=[]
    for m in (frame or {}).get('measures',[]):
        if m not in [x['value'] for x in measures]:measures.append(dict(label=label(m),value=copy.deepcopy(m)))
    labels={r['id']:r.get('short') or r['id'].upper() for r in regions}
    ids=list(c.get('regions') or [])
    try:
        from query_semantics import _samples_sources,_possible,_region_aliases
        sources=_samples_sources(tree);allowed=set(labels)
        for source in sources:allowed &= _possible(where.this if where else None,'region_id',_region_aliases(tree,source.alias_or_name),set(labels))
        if sources and allowed!=set(labels):ids=sorted(allowed)
        elif not sources and not ids:ids=list((history.get('population') or {}).get('region_ids') or [])
    except (TypeError,ValueError):pass
    shown=[]
    for row in history.get('rows') or []:
        for key in ('region_id','region_name'):
            value=str(row.get(key,''))
            found=region_ids(value,regions)
            for rid in found:
                if rid not in shown:shown.append(rid)
    filters=[]
    if where:
        def required_terms(node):
            if isinstance(node,exp.Paren):return required_terms(node.this)
            if isinstance(node,exp.And):return required_terms(node.this)+required_terms(node.expression)
            return [node]
        field_labels={'reviewed':{0:'未复核',1:'已复核'},'has_holes':{1:'有空洞标记',0:'无空洞标记'},'has_fragments':{1:'有碎片标记',0:'无碎片标记'},'review_level':{'high':'高风险','medium':'中风险','low':'低风险','none':'无风险标记','unknown':'风险状态未知'}}
        for node in required_terms(where.this):
            if not isinstance(node,exp.EQ):continue
            a,b=node.this,node.expression
            if isinstance(a,exp.Literal):a,b=b,a
            if isinstance(a,exp.Column) and isinstance(b,exp.Literal) and a.name in field_labels:
                label_value=field_labels[a.name].get(int(b.this) if str(b.this).isdigit() else b.this)
                if label_value and label_value not in filters:filters.append(label_value)
    return dict(schemaVersion=1,source='executed_sql',regions=ids,regionLabels=[labels.get(r,r) for r in ids] or ['全部区域'],
        resultRegions=shown,measures=measures,filters=filters,filterExpression=where.this.sql(dialect='sqlite') if where else None,
        grouping=copy.deepcopy((frame or {}).get('grouping',[])),regionSelection=copy.deepcopy((frame or {}).get('regionSelection')),
        order=[dict(expression=o.this.sql(dialect='sqlite'),direction='DESC' if o.args.get('desc') else 'ASC') for o in (tree.args.get('order') or exp.Order()).expressions],
        limit=c.get('limit'),offset=c.get('offset',0),sampleCount=(history.get('population') or {}).get('sample_count'))


def public_pending(history):
    pending=(history or {}).get('pending') or {};record=pending.get('decision')
    if not isinstance(record,dict):return None
    return {key:copy.deepcopy(record[key]) for key in ('id','kind','slot','question','options','task','acceptsText')}


def make_pending(history,question,message,slot,options,version,regions,*,kind='user_choice',family=None):
    """Option values remain private to the version-bound server history."""
    current=semantic_state(history,regions) or {}
    ids=region_ids(question,regions)
    labels={r['id']:r.get('short') or r['id'].upper() for r in regions}
    task=dict(regions=ids or current.get('regions',[]),regionLabels=[labels[r] for r in ids] if ids else current.get('regionLabels',['全部区域']),
        measures=[CATALOG[family]['label']+'统计口径待确认'] if family else [m['label'] for m in current.get('measures',[])],filters=current.get('filters',[]))
    if slot=='risk_level':task['measures']=['风险等级待确认']
    if slot=='region' and re.search(r'样本.{0,8}(?:多少|数量|总数)|多少样本',question):task['measures']=['样本数量']
    if ids and set(ids)!=set(current.get('regions',[])):task['filters']=[]
    decision=dict(id=uuid.uuid4().hex,kind=kind,slot=slot,question=message,options=[],task=task,acceptsText=kind=='user_choice' and not options,version=version)
    values={}
    for i,option in enumerate(options):
        oid='option_'+str(i+1)
        decision['options'].append(dict(id=oid,label=option['label'],description=option.get('description','')))
        values[oid]=copy.deepcopy(option['value'])
    return dict(question=question,turns=[],clarification=message,metricChoice=False,decision=decision,choiceValues=values)


def _ordinal_reply(question):
    match=re.fullmatch(r'\s*(?:(?:我)?(?:选择|选|要|看)|就选|先看)?\s*(?:第)?([1-9一二两三四五六七八九])(?:个|种|项)?(?:吧)?[。！!\s]*',question)
    if not match:return None
    return int(match[1]) if match[1].isdigit() else '一二三四五六七八九'.find('二' if match[1]=='两' else match[1])+1


def matching_choice(pending,question):
    options=(pending.get('decision') or {}).get('options',[])
    ordinal=_ordinal_reply(question)
    if ordinal is not None:return options[ordinal-1]['id'] if 1<=ordinal<=len(options) else None
    text=question.strip(' 。！!')
    for option in options:
        value=pending.get('choiceValues',{}).get(option['id'],{})
        if text==option['label'] or text in value.get('replies',[]):return option['id']
    return None


def can_resolve_text(query_context,question):
    pending=(query_context or {}).get('pendingRecord') or {}
    record=pending.get('decision') or {}
    return bool(record and _ordinal_reply(question) is not None or matching_choice(pending,question) or record.get('acceptsText') and record.get('slot')=='threshold' and re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',question))


def _apply(pending,oid):
    value=copy.deepcopy(pending['choiceValues'][oid]);original=pending['question']
    if value['type']=='segmentation_metric':
        question=original.rstrip('。？?')+'；更正原问题，改为'+value['question']+'；仅保留原问题明确的区域和筛选，不再统计变化或建筑栋数。'
    elif value['type']=='metric':
        question=original.rstrip('。？?')+'；明确统计口径：'+value['definition']+'。'
    elif value['type']=='region':
        question=re.sub(r'这个区域|该区域|这个地区|该地区|刚才那个区域',value['region'].upper()+'区域',original)
        if question==original:question=original.rstrip('。？?')+'；本轮查询区域：'+value['region'].upper()+'。'
    elif value['type']=='risk':
        question=original.replace('风险',value['label'],1)
    elif value['type']=='sort':
        question=original.rstrip('。？?')+'；排序指标：'+value['label']+'；按该指标'+('降序' if value['direction']=='DESC' else '升序')+'排列，其余统计条件不变。'
    elif value['type']=='comparison_measure':
        question=original.rstrip('。？?')+'；明确比较指标：'+value['label']+'。'
    else:raise ValueError('Unsupported clarification update')
    return dict(question=question,binding=value,clarificationResolution=dict(id=pending['decision']['id'],optionId=oid,slot=pending['decision']['slot'],label=value.get('definition') or value.get('label') or value.get('region')))


def resolve_reply(history,payload,question,version):
    pending=(history or {}).get('pending') or {};record=pending.get('decision') or {}
    cid=payload.get('clarificationId');oid=payload.get('optionId')
    if cid is not None or oid is not None:
        if not isinstance(cid,str) or not isinstance(oid,str) or cid!=record.get('id') or record.get('version')!=version or oid not in pending.get('choiceValues',{}):
            return dict(error='这条选择已经失效，请使用当前待确认问题的选项，或重新描述查询。',source='stale_clarification')
        return _apply(pending,oid)
    if record and record.get('version')==version:
        if record.get('acceptsText') and re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',question) and record['slot']=='threshold':
            return dict(question=pending['question'].rstrip('。？?')+'；用户明确给出的阈值：'+question.strip()+'。',binding=dict(type='threshold'),clarificationResolution=dict(id=record['id'],slot='threshold',label=question.strip()))
        matched=matching_choice(pending,question)
        if matched:return _apply(pending,matched)
        if _ordinal_reply(question) is not None:
            return dict(error='这个序号不在当前选项中，请选择下方有效选项。',source='invalid_clarification_choice')
        supplemental=bool(record.get('acceptsText') and re.match(r'^(?:用|按|分母|分子|阈值|口径|统计口径)',question) and not re.search(r'新问题|换个问题|另外查询|解释|什么是',question))
        if supplemental:
            return dict(question=pending['question'].rstrip('。？?')+'；用户补充条件：'+question+'。',binding=dict(type='supplement'),clarificationResolution=dict(id=record['id'],slot=record['slot'],label=question))
    return None


def metric_options(family):
    item=CATALOG[family]
    return [
        dict(label='标记样本占比',description=item['marker']+'的样本数 ÷ 当前范围全部样本数',value=dict(type='metric',family=family,field=item['flag'],operation='flag_percentage',definition=item['marker']+'的样本占当前统计范围全部样本的百分比',replies=['样本占比','看样本占比',item['marker']+'的样本占比'])),
        dict(label='面积比例均值',description='各样本'+item['area']+'的平均值',value=dict(type='metric',family=family,field=item['field'],operation='avg',definition='每张样本'+item['area']+item['field']+'的均值',replies=['面积比例均值','看面积比例','平均面积比例',item['field']])),
    ]


def _frame(history):
    from query_dialogue import task_frame
    return task_frame(history) or {}


def _sort_options(history,question):
    from query_dialogue import label
    frame=_frame(history);measures=frame.get('measures',[])
    if len((history or {}).get('rows',[]))<2:return []
    if not measures or not re.search(r'排序|排一下|排个序|从高到低|从低到高|从大到小|从小到大',question):return []
    if re.search(r'数量|总数|占比|百分|比例|均值|平均|像素|对象|风险|空洞|碎片',question):return []
    # A previously explicit metric order is a usable continuation. Region-name
    # presentation order is not a metric choice.
    try:
        tree=sqlglot.parse_one(history['sql'],read='sqlite');orders=(tree.args.get('order') or exp.Order()).expressions
        if orders and (history.get('queryContract') or {}).get('orderRequired',True):
            target=orders[0].this
            if isinstance(target,exp.Column) and not target.table:target=next((p.this for p in tree.expressions if isinstance(p,exp.Alias) and p.alias==target.name),target)
            if target.find(exp.AggFunc):return []
    except (ValueError,sqlglot.errors.ParseError):pass
    if len(measures)<2:return []
    direction='DESC' if re.search(r'从高到低|从大到小|降序',question) else 'ASC' if re.search(r'从低到高|从小到大|升序',question) else None
    options=[]
    for m in measures:
        for d in ([direction] if direction else ['DESC','ASC']):
            options.append(dict(label=label(m)+('从高到低' if d=='DESC' else '从低到高'),value=dict(type='sort',label=label(m),measure=m,direction=d)))
    return options


def detect(history,question,regions,scope,version):
    """Return a decision only for known business ambiguities, not model errors."""
    frame=_frame(history)
    row_request=bool(re.search(r'样本ID|sample_id|列出|每张|单张|[0-9一二两三四五六七八九十]+\s*张|(?:最高|最大|最多|最低|最小).{0,8}样本',question,re.I))
    for family,item in CATALOG.items():
        mentions=any(name in question for name in item['names'])
        if not mentions or row_request or not re.search(r'比例|占比|率',question):continue
        explicit=bool(re.search(r'平均|均值|面积|'+re.escape(item['field']),question,re.I) or re.search(r'(?:有|带).{0,4}'+re.escape(item['label'])+r'(?:标记)?(?:的)?样本.{0,4}(?:占比|比例|百分)',question) or (item['marker'] in question and '样本' in question))
        if explicit:continue
        old=frame.get('measures',[])
        matching=[m for m in old if m['field'] in {item['flag'],item['field']}]
        if matching and len(matching)==len(old):continue
        options=metric_options(family)
        message='这里的“'+item['label']+'比例”要按哪种口径计算？范围和其他明确条件会保留。'
        return make_pending(history,question,message,'measure_definition',options,version,regions,family=family)
    # Bare "风险占比" has no single business definition in this schema.
    if re.search(r'风险.{0,6}(?:占比|比例|数量|多少)',question) and not re.search(r'[高中低小]风险',question) and not any(m.get('risk') for m in frame.get('measures',[])):
        options=[dict(label=label,value=dict(type='risk',label=label,replies=[label])) for label in ('高风险','中风险','低风险')]
        return make_pending(history,question,'要统计哪个风险等级？','risk_level',options,version,regions)
    pronoun=unresolved_region_reference(question,regions)
    if pronoun:
        aliases={a.lower():r['id'] for r in regions for a in (r['id'],r.get('short',''),r.get('name','')) if a}
        available=[]
        for row in (history or {}).get('rows',[]):
            for key in ('region_id','region_name'):
                rid=aliases.get(str(row.get(key,'')).lower())
                if rid and rid not in available:available.append(rid)
        if len(available)!=1:
            candidates=[r for r in regions if (not scope or r['id']==scope) and (not available or r['id'] in available)]
            options=[dict(label=r.get('short') or r['id'].upper(),value=dict(type='region',region=r['id'],replies=[r['id'],r['id'].upper()])) for r in candidates]
            return make_pending(history,question,'上一轮没有唯一确定一个区域，要继续查看哪个？','region',options,version,regions)
    options=_sort_options(history,question)
    if options:return make_pending(history,question,'这份结果包含多个统计指标，要按哪一个排序？','sort_metric',options,version,regions)
    return None


def model_pending(history,question,message,missing,version,regions,scope):
    """Model missing-slot output shares the same stored protocol as catalog choices."""
    slot=missing.get('field','business_definition');kind=missing.get('kind','user_choice');options=[]
    if kind=='user_choice' and slot=='region':
        options=[dict(label=r.get('short') or r['id'].upper(),value=dict(type='region',region=r['id'],replies=[r['id'],r['id'].upper()])) for r in regions if not scope or r['id']==scope]
    if kind=='user_choice' and slot=='sort_metric':options=_sort_options(history,'排个序')
    return make_pending(history,question,message,slot,options,version,regions,kind=kind)


def problem(message,source):
    """Non-interactive limits use the same shape without creating fake choices."""
    return dict(id=None,kind='context_required',slot=source,question=message,options=[],task={},acceptsText=False)


def sql_missing_slot(message,known_columns=()):
    """SQL syntax/formatting failures are internal repairs, never user choices."""
    if re.search(r'语法|syntax|ambiguous\s+column|别名|输出列|查询要求.{0,8}返回|\bJSON\b|\b(?:ORDER\s+BY|GROUP\s+BY|ROW_NUMBER|CTE|SELECT|WHERE|JOIN|LIMIT|OFFSET)\b|SQL.{0,10}(?:校验|不符合|格式)|请提供更多|需要更多信息',message,re.I):return None
    if re.search(r'字段|column|field|constraint|missing|not include|required|缺少|缺失|不存在',message,re.I) and any(re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',message,re.I) for name in known_columns):return None
    if re.search(r'缺少|缺失|没有|未提供|不包含|不存在',message) and re.search(r'日期|拍摄|采集|真值|平方米|地面尺度|分辨率',message):
        return dict(kind='unavailable_field',field='data_field')
    field='threshold' if re.search(r'阈值|门槛',message) else 'region' if re.search(r'哪个区域|指定区域|选择区域',message) else 'sort_metric' if re.search(r'哪个指标.{0,6}排序|排序指标',message) else 'business_definition'
    return dict(kind='user_choice',field=field)
