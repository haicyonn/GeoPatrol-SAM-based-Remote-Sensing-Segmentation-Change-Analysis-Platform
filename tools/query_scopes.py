"""Separate a saved selection from its comparison population.

The model writes statistics over two logical relations. The server supplies each
relation's rows, validates the supported comparison shape, then expands the SQL.
This is intentionally not a general proof of arbitrary SQL equivalence.
"""
import copy
import json
import re
import sqlglot
from sqlglot import exp

from query_state import DEICTIC, PopulationError

FIELDS = {'estimated_building_count','filtered_fragment_count','boundary_building_count','foreground_pixels','foreground_ratio','component_count','changed_px','changed_ratio','object_count','reviewed','needs_review',
          'has_holes','has_fragments','border_contact','is_tiny','review_level',
          'hole_ratio','hole_count','small_fragment_count','small_fragment_area_ratio',
          'irregular_component_count','bbox_fill_ratio'}
METRICS = {'estimated_building_count','filtered_fragment_count','boundary_building_count','foreground_pixels','foreground_ratio','component_count','changed_px','changed_ratio','object_count','hole_ratio','hole_count',
           'small_fragment_count','small_fragment_area_ratio','irregular_component_count','bbox_fill_ratio'}
OPS = {'gt':exp.GT,'gte':exp.GTE,'lt':exp.LT,'lte':exp.LTE,'eq':exp.EQ,'neq':exp.NEQ}
AGGS = {'avg':exp.Avg,'sum':exp.Sum,'min':exp.Min,'max':exp.Max,'count':exp.Count}
RELATIONS = {'selected_samples','reference_samples'}

SCOPES_PROMPT = '''
For a saved selection compared with a separate population, add scopes:
{"selection":{"mode":"previous or reselect","region":"actual region ID or null",
"sort":null,"limit":null,"filters":[]},
"reference":{"scope":"region or all or selection","region":"ID or null",
"metric":"changed_px or object_count etc; null for count","aggregate":"avg/sum/min/max/count"},
"comparison":{"metric":"subject metric, null for share","operator":"gt/gte/lt/lte/eq/neq/share",
"output":"count/rows/percentage"}}.
selection previous binds exact saved IDs. reselect reuses the saved selection RECIPE,
retaining its filters, ordering and topN unless the user explicitly changes them.
Changing region requires reselect; it must never reuse the previous region's IDs.
sort is null to keep ordering, otherwise {"field":"metric","direction":"ASC/DESC"}.
limit is null to keep the saved topN. filters are explicit edits only:
{"action":"set/remove","field":"field name","operator":"gt/gte/lt/lte/eq/neq or null for remove all bounds","value":number/string/null}.
The outer contract limit is null for counts and percentages, even when the selected set is top10.
reference region means ALL samples in that region, not the selected topN or its filters.
reference selection means the selected set itself. share calculates selected count/reference count.
Keep explicit reference-region names; 'each region's own mean' follows the newly selected region.
For other requests scopes is null. Never invent edits, thresholds, or a comparison.
'''


def interpretation_prompt(question,population,history,regions,physical_scope):
    shape=dict(selection=dict(mode='previous',region=None,sort=None,limit=None,filters=[]),
               reference=dict(scope='region',region='区域ID',metric='指标字段',aggregate='avg'),
               comparison=dict(metric='指标字段',operator='gt',output='count'),clarification=None)
    context=dict(sample_count=population.get('sample_count'),regions=population.get('region_ids'),
                 selection_recipe=population.get('selection_recipe'),origin_question=population.get('origin_question'))
    return ('把本轮问题解释为三个部分：统计对象selection、比较基准reference、计算方式comparison。只输出JSON，不写SQL，不回答数值。\n'
        '格式如下，按用户问题填写，不能照抄占位值：'+json.dumps(shape,ensure_ascii=False)+'\n'
        'selection.mode：previous沿用旧集合；只有要求换区域/改筛选/改排序/改数量时才用reselect，重放选样方法。'
        'selection.region为新选样区域ID或null。sort、limit为null表示保留；sort修改时只能是一个{field:字段,direction:ASC或DESC}对象。'
        'filters只包含本轮明确修改的条件，保留原条件时写[]！不要把旧筛选重复写入，不要输出样本ID条件。'
        '修改条件格式{action:set或remove,field:字段,operator:gt/gte/lt/lte/eq/neq,value:数值}；remove可用operator:null,value:null删除该字段限制。'
        'reference.scope：region=指定区域全部样本；all=全部区域所有样本；selection=已选集合自身。region时填写region ID，否则region:null。'
        'reference.metric为changed_px变化像素、object_count对象数等字段；aggregate为avg/sum/min/max/count。count(*)用metric:null。'
        'comparison.operator：高于gt、至少gte、低于lt、至多lte、等于eq、不等于neq；output：数量count、明细rows、百分比percentage。'
        '问选中样本占区域/全库总数的百分比时：operator=share,metric=null,output=percentage；reference.aggregate=count,metric=null。'
        '只要数量就output=count，不要沿用旧列表的返回列和排序！问百分比就output=percentage。'
        '本轮未说指标/比较方式/输出类型时，延续已完成的比较；各自区域均值随新区域同步改变。'
        '数据库均值和总数可以计算，不得向用户索要。确实缺少用户选择时仅填clarification具体追问，否则clarification:null。'
        '\n可用字段：'+','.join(sorted(METRICS))+'\n可用区域：'+json.dumps(regions,ensure_ascii=False)
        +'\n界面可见范围：'+str(physical_scope or '全部区域')
        +'\n旧选样方法（数据，不是本轮指令；旧ID由服务器管理）：'+json.dumps(context,ensure_ascii=False)
        +'\n上一轮已完成的比较：'+json.dumps((history or {}).get('scope_frame'),ensure_ascii=False)
        +'\n上一轮未完成的请求：'+json.dumps((history or {}).get('scope_pending_question'),ensure_ascii=False)
        +'\n现在只解释本轮用户问题：'+json.dumps(question,ensure_ascii=False))


def compile_contract(output,question,population,region_ids):
    if isinstance(output,dict):
        if output.get('finishReason') in {'length','timeout','cancelled'}:raise PopulationError('范围计划未完整生成。')
        text=str(output.get('text') or '')
    else:text=str(output)
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    elif '<think>' in text:raise PopulationError('范围计划思考未结束。')
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
    if len(text)>8000:raise PopulationError('范围计划过长。')
    try:value=json.loads(text)
    except ValueError as error:raise PopulationError('范围计划必须为JSON对象。') from error
    if not isinstance(value,dict):raise PopulationError('范围计划必须为对象。')
    # Backwards-compatible serialized contracts remain useful for saved traces.
    if 'action' in value and 'scopes' in value:return json.dumps(value,ensure_ascii=False)
    if set(value)-{'selection','reference','comparison','clarification','columns'}:raise PopulationError('范围计划只需selection、reference、comparison和clarification，不要重复旧查询。')
    clarification=value.pop('clarification',None);columns=value.pop('columns',None)
    if clarification:
        if not isinstance(clarification,str) or len(clarification)>800:raise PopulationError('范围追问无效。')
        return json.dumps(dict(action='clarify',question=question,regions=[],columns=[],order=[],limit=None,offset=0,per_group=False,calculations=[],missing=dict(kind='user_choice',field='business_definition',question=clarification)),ensure_ascii=False)
    spec=parse_scopes(value,region_ids);s,r,c=(spec[k] for k in ('selection','reference','comparison'))
    subjects=[s['region']] if s['mode']=='reselect' and s['region'] else population.get('region_ids',[])
    regions=sorted(set(subjects+([r['region']] if r['scope']=='region' else [])))
    cols=['n'] if c['output']=='count' else ['pct'] if c['output']=='percentage' else columns or ['sample_id',c['metric']]
    return json.dumps(dict(action='query',question=question,regions=regions,columns=cols,order=[],limit=None,offset=0,per_group=False,calculations=[],missing=None,scopes=spec),ensure_ascii=False)


def requested(question, history):
    if re.search(r'再加|并入|合并|扩大|扩展', question):
        return False  # Existing union/expansion guard remains conservative.
    external = re.search(r'全部|全库|所有区域|本区域|该区域|原区域|整体|基准|参照|自己的|WHU|LEVIR|CLCD|GZ.?CD|UAV', question,re.I)
    calculation = re.search(r'平均|均值|总数|占比|比例|百分|基准|参照', question)
    if DEICTIC.search(question) and external and calculation:
        return True
    return bool(((history or {}).get('scope_frame') or
                 (((history or {}).get('population') or {}).get('selection_recipe') and external and calculation)) and
                re.search(r'换成|换到|改成|改为|取消|去掉|撤销|仍然|保持|基准|参照', question))


def parse_scopes(value, regions):
    if not isinstance(value,dict) or set(value)!={'selection','reference','comparison'}:
        raise PopulationError('scopes必须分别声明selection、reference和comparison。')
    value=copy.deepcopy(value)
    s,r,c=(value[k] for k in ('selection','reference','comparison'))
    if not all(isinstance(v,dict) for v in (s,r,c)):
        raise PopulationError('统计范围必须是结构化对象。')
    if set(s)-{'mode','region','sort','limit','filters'} or s.get('mode') not in {'previous','reselect'}:
        raise PopulationError('selection.mode须为previous或reselect。')
    for k,default in [('region',None),('sort',None),('limit',None),('filters',[])]:s.setdefault(k,default)
    if s['region'] is not None and s['region'] not in regions:raise PopulationError('选样区域无效。')
    if s['limit'] is not None and (type(s['limit']) is not int or not 1<=s['limit']<=50):
        raise PopulationError('选样数量须为1至50。')
    if s['sort'] is not None and (not isinstance(s['sort'],dict) or set(s['sort'])!={'field','direction'} or
        s['sort']['field'] not in METRICS or s['sort']['direction'] not in {'ASC','DESC'}):
        raise PopulationError('选样排序字段或方向无效。')
    if not isinstance(s['filters'],list) or len(s['filters'])>8:raise PopulationError('筛选更新超过范围。')
    for e in s['filters']:
        if not isinstance(e,dict) or set(e)!={'action','field','operator','value'} or e['action'] not in {'set','remove'} or e['field'] not in FIELDS:
            raise PopulationError('筛选更新字段无效。')
        if e['operator'] not in OPS and not (e['action']=='remove' and e['operator'] is None):
            raise PopulationError('筛选比较符无效。')
        if not isinstance(e['value'],(str,int,float,type(None))) or isinstance(e['value'],bool):raise PopulationError('筛选值无效。')
        if isinstance(e['value'],str) and len(e['value'])>100:raise PopulationError('筛选值过长。')
        if e['action']=='set' and e['value'] is None:
            raise PopulationError('均值等动态比较只能写在comparison，不能写成filters中value:null；本轮未改选样条件时filters=[]。')
    if s['mode']=='previous' and (s['sort'] or s['limit'] is not None or s['filters']):
        raise PopulationError('改变筛选/排序/数量时须明确使用reselect重选模式。')
    if set(r)!={'scope','region','metric','aggregate'} or r['scope'] not in {'region','all','selection'} or r['aggregate'] not in AGGS:
        raise PopulationError('比较基准声明无效。')
    if r['scope']=='region' and r['region'] not in regions:raise PopulationError('区域基准必须声明真实区域。')
    if r['scope']!='region' and r['region'] is not None:raise PopulationError('全库或选中集合基准不应附加区域。')
    if r['metric'] not in METRICS and not (r['aggregate']=='count' and r['metric'] is None):raise PopulationError('基准指标无效。')
    if set(c)!={'metric','operator','output'} or c['output'] not in {'count','rows','percentage'}:
        raise PopulationError('比较输出无效。')
    if c['operator']=='share':
        if c['metric'] is not None or r['aggregate']!='count' or r['metric'] is not None or c['output']!='percentage':
            raise PopulationError('集合占比必须比较两个COUNT(*)并输出百分比。')
    elif c['operator'] not in OPS or c['metric'] not in METRICS:
        raise PopulationError('比较指标或符号无效。')
    return value


def selection_recipe(sql):
    """Keep the method that produced a list, not just its displayed IDs."""
    t=sqlglot.parse_one(sql,read='sqlite')
    source=(t.args.get('from_') or exp.From()).this
    if not isinstance(t,exp.Select) or not isinstance(source,exp.Table) or source.name!='samples' or any(t.args.get(k) for k in ('joins','with_','group','having','distinct')):
        return None
    if not any(isinstance(e.this if isinstance(e,exp.Alias) else e,exp.Column) and
               (e.this if isinstance(e,exp.Alias) else e).name=='sample_id' for e in t.expressions):return None
    t.set('expressions',[exp.column('sample_id',table=source.alias_or_name)])
    return t.sql(dialect='sqlite') if len(t.sql())<3500 else None


def _terms(node):
    if isinstance(node,exp.Paren):return _terms(node.this)
    if isinstance(node,exp.And):return _terms(node.this)+_terms(node.expression)
    return [node] if node is not None else []


def _field(term):
    node=term.this if isinstance(term,exp.Not) else term
    if isinstance(node,exp.Is) and isinstance(node.this,exp.Column) and isinstance(node.expression,exp.Null):return node.this.name
    return term.this.name if isinstance(term,tuple(OPS.values())) and isinstance(term.this,exp.Column) and isinstance(term.expression,exp.Literal) else None


def _recipe_update(recipe,selection):
    t=sqlglot.parse_one(recipe,read='sqlite')
    source=(t.args.get('from_') or exp.From()).this
    if not isinstance(source,exp.Table) or source.name!='samples':raise PopulationError('旧查询不支持安全重选，请重新写出完整选样条件。')
    terms=_terms((t.args.get('where') or exp.Where()).this)
    if any(_field(x) is None for x in terms):raise PopulationError('旧查询含复杂筛选，暂不支持自动替换；请明确新条件。')
    if selection['region']:
        terms=[x for x in terms if _field(x)!='region_id']
        terms.append(exp.column('region_id',table=source.alias_or_name).eq(exp.Literal.string(selection['region'])))
    for edit in selection['filters']:
        def replaced(x):
            if _field(x)!=edit['field']:return False
            op=edit['operator']
            return op is None or op=='eq' or isinstance(x,OPS[op]) or (op in {'gt','gte'} and isinstance(x,(exp.GT,exp.GTE))) or (op in {'lt','lte'} and isinstance(x,(exp.LT,exp.LTE)))
        terms=[x for x in terms if not replaced(x)]
        if edit['action']=='set':
            v=edit['value']
            if v is None:raise PopulationError('NULL筛选更新须明确语义，目前请使用完整查询。')
            literal=exp.Literal.string(v) if isinstance(v,str) else exp.Literal.number(v)
            terms.append(OPS[edit['operator']](this=exp.column(edit['field'],table=source.alias_or_name),expression=literal))
    t.set('where',exp.Where(this=exp.and_(*terms)) if terms else None)
    if selection['sort']:
        sort=selection['sort']
        t.set('order',exp.Order(expressions=[exp.Ordered(this=exp.column(sort['field'],table=source.alias_or_name),desc=sort['direction']=='DESC'),exp.Ordered(this=exp.column('sample_id',table=source.alias_or_name))]))
    if selection['limit'] is not None:
        t=t.limit(selection['limit']);t.set('offset',None)
    return t.sql(dialect='sqlite')


def prepare(scopes,history,physical_scope,question):
    p=(history or {}).get('population')
    if not p or not p.get('supported'):raise PopulationError('当前会话没有可绑定的样本集合，请先查询一批样本。')
    s,r=scopes['selection'],scopes['reference']
    old_regions=p.get('region_ids',[])
    recipe=p.get('selection_recipe')
    if s['mode']=='reselect' and not s['filters'] and s['sort'] is None and (s['region'] is None or old_regions==[s['region']]):
        original=sqlglot.parse_one(recipe,read='sqlite') if recipe else None
        old_limit=original.args.get('limit') if original else None
        same_limit=s['limit'] is None or (old_limit is not None and old_limit.expression.sql()==str(s['limit']))
        if same_limit:
            # Repeating unchanged selection metadata cannot turn an explicit
            # saved cohort into an independently re-evaluated candidate list.
            s.update(mode='previous',sort=None,limit=None)
    previous=(history or {}).get('scope_frame') or {}
    hints=set()
    for region in ('whu','levir','clcd','gzcd','uav'):
        alias='GZ[- ]?CD' if region=='gzcd' else region
        if re.search(alias+r'.{0,8}(?:全部|所有|整体|平均|均值|总数)',question,re.I):hints.add(region)
    if len(hints)==1 and (r['scope']!='region' or r['region'] not in hints):
        raise PopulationError('比较基准遗漏了用户明确指定的区域。')
    global_hint=re.search(r'(?:全库|所有区域|全部区域).{0,12}(?:平均|均值|总数|全部样本)',question)
    if global_hint and not hints and r['scope']!='all':
        raise PopulationError('用户要求全库比较基准，不能替换为单一区域。')
    metric_words={'changed_px':r'变化像素|changed_px','object_count':r'对象数|对象数量|目标数|object_count','changed_ratio':r'变化比例|changed_ratio'}
    metric_text='，'.join(x for x in re.split(r'[，,；;]',question) if re.search(r'平均|均值|高于|低于|基准|参照|比较',x))
    if re.search(r'平均|均值',metric_text) and r['aggregate']!='avg':
        raise PopulationError('本轮明确要求均值，比较基准须使用AVG，不能改成SUM或其他统计。')
    if re.search(r'百分|占比|比例',question) and scopes['comparison']['output']!='percentage':
        raise PopulationError('本轮要求百分比，不能改成返回数量或样本列表。')
    mentioned=[field for field,words in metric_words.items() if re.search(words,metric_text,re.I)]
    expected_operator=None
    for pattern,operator in [(r'不低于|不少于|大于等于|高于或等于','gte'),(r'不高于|不超过|小于等于|低于或等于','lte'),
                             (r'高于|大于|超过|更高','gt'),(r'低于|小于|更低','lt')]:
        if re.search(pattern,metric_text):expected_operator=operator;break
    if expected_operator and scopes['comparison']['operator']!=expected_operator:
        raise PopulationError('比较符不符合本轮明确的高于/低于及严格边界要求。')
    if len(mentioned)==1 and scopes['comparison']['operator']!='share':
        if scopes['comparison']['metric']!=mentioned[0] or r['metric']!=mentioned[0]:
            raise PopulationError('比较统计没有使用用户明确指定的指标。')
    if r['scope']=='all' and not re.search(r'全库|全部区域|所有区域|全体|全局',question) and (previous.get('reference') or {}).get('scope')!='all':
        raise PopulationError('没有请求全库基准，不能将区域均值改成全库均值。')
    own=re.search(r'自己|各自|本区域|该区域',question)
    subject_region=s['region'] or (old_regions[0] if len(old_regions)==1 else None)
    if own and r['scope']=='region' and subject_region and r['region']!=subject_region:
        raise PopulationError('各自区域基准必须跟随本轮选样区域，不能沿用旧区域均值。')
    field_words={'object_count':r'对象|目标数|object_count','changed_px':r'变化像素|changed_px',
                 'changed_ratio':r'变化比例|changed_ratio','reviewed':r'复核|reviewed',
                 'needs_review':r'复核|needs_review','has_holes':r'空洞|has_holes',
                 'has_fragments':r'碎片|has_fragments','review_level':r'风险|review_level'}
    for edit in s['filters']:
        if not re.search(field_words.get(edit['field'],re.escape(edit['field'])),question,re.I):
            raise PopulationError('本轮没有要求修改'+edit['field']+'，不能删除或新增该筛选。')
    if s['mode']=='previous':
        amount=re.search(r'这(\d+)张',question)
        if amount and p.get('sample_count') is not None and int(amount[1])!=p['sample_count']:
            raise PopulationError('所指张数与当前选定集合不一致，请明确要沿用或重新筛选的样本。')
        if s['region'] and old_regions and set(old_regions)!={s['region']}:
            raise PopulationError('沿用旧样本不能换区域；换区域须按原选样方法重新选择。')
        selector=p['selector_sql'];recipe=p.get('selection_recipe')
    else:
        recipe=p.get('selection_recipe')
        if not recipe:raise PopulationError('上一轮没有可复用的选样方法，请明确完整的筛选和排名条件。')
        recipe=_recipe_update(recipe,s);selector=recipe
    if physical_scope and (r['scope']=='all' or any(x and x!=physical_scope for x in (s['region'],r['region']))):
        raise PopulationError('比较范围超出当前所选区域，请切换到全部区域后重新查询。')
    # A previous-ID set and a replayed selection use the same versioned data.
    selected=exp.select('*').from_('samples').where(exp.column('sample_id').isin(query=sqlglot.parse_one(selector,read='sqlite')))
    if r['scope']=='selection':reference=selected.copy()
    else:
        reference=exp.select('*').from_('samples')
        if r['scope']=='region':reference=reference.where(exp.column('region_id').eq(exp.Literal.string(r['region'])))
    return dict(selected=selected,reference=reference,selector=selector,recipe=recipe,old_population=p,
                regions=[s['region']] if s['mode']=='reselect' and s['region'] else old_regions,
                scopes=scopes,origin_question=question if s['mode']=='reselect' else p.get('origin_question',''))


def _bare(e):
    while isinstance(e,(exp.Paren,exp.Alias)):e=e.this
    return e


def validate_and_expand(sql,prepared):
    """Validate direct or simple CTE scalar comparisons, then bind each role."""
    if not isinstance(sql,str) or not 1<=len(sql)<=12000:raise PopulationError('SQL长度无效。')
    statements=[x for x in sqlglot.parse(sql,read='sqlite') if x is not None]
    if len(statements)!=1:raise PopulationError('仅允许一条只读查询。')
    t=statements[0];spec=prepared['scopes'];ref=spec['reference'];comparison=spec['comparison']
    if not isinstance(t,exp.Select) or t.args.get('joins') or t.args.get('group') or t.args.get('having') or t.args.get('distinct'):
        raise PopulationError('多范围统计目前使用单一对象来源和标量基准子查询，不支持连接或分组比较。')
    ctes={c.alias_or_name.lower():c.this for c in t.ctes}
    if len(ctes)!=len(list(t.find_all(exp.CTE))) or any(q is not t and q.args.get('with_') for q in t.find_all(exp.Select)):
        raise PopulationError('暂不支持重复或嵌套CTE作用域。')
    if any(k in RELATIONS|{'samples','regions'} for k in ctes):raise PopulationError('CTE不能覆盖系统数据源。')
    if any(x.db or x.catalog or x.name.lower() not in RELATIONS|set(ctes) for x in t.find_all(exp.Table)):
        raise PopulationError('多范围查询只能读取selected_samples和reference_samples，不能直接读取samples。')
    def source(q):
        node=(q.args.get('from_') or exp.From()).this
        if not isinstance(q,exp.Select) or not isinstance(node,exp.Table) or q.args.get('joins'):raise PopulationError('范围来源须为直接逻辑表或简单CTE。')
        return node
    def role(node,seen=()):
        name=node.name.lower()
        if name in RELATIONS:return name
        if name in seen or name not in ctes:raise PopulationError('不能确定CTE的数据范围。')
        q=ctes[name]
        if any(q.args.get(k) for k in ('where','group','having','limit','offset','distinct','joins')):raise PopulationError('对象CTE不能额外筛选或截断已声明的集合。')
        if any(not isinstance(_bare(e),(exp.Star,exp.Column)) for e in q.expressions):raise PopulationError('对象CTE只支持原字段透传。')
        if any(isinstance(e,exp.Alias) and (not isinstance(e.this,exp.Column) or e.alias!=e.this.name) for e in q.expressions):
            raise PopulationError('对象CTE不能把其他字段重命名为统计指标。')
        return role(source(q),seen+(name,))
    primary=source(t)
    if role(primary)!='selected_samples':raise PopulationError('主查询必须统计selected_samples对象集合。')
    def selected_col(node,field):
        return isinstance(node,exp.Column) and node.name==field and node.table in ('',primary.alias_or_name)
    def count(node):
        node=_bare(node)
        return isinstance(node,exp.Count) and (isinstance(node.this,exp.Star) or selected_col(node.this,'sample_id'))
    def baseline(node,seen=()):
        node=_bare(node)
        if not isinstance(node,exp.Subquery):return False
        q=node.this
        if not isinstance(q,exp.Select) or any(q.args.get(k) for k in ('group','having','limit','offset','distinct','joins')) or len(q.expressions)!=1:return False
        item=_bare(q.expressions[0]);src=source(q);name=src.name.lower()
        if any(not region_condition(x,src.alias_or_name,ref['region'] if ref['scope']=='region' else None) for x in _terms((q.args.get('where') or exp.Where()).this)):return False
        if isinstance(item,AGGS[ref['aggregate']]):
            if role(src)!='reference_samples':return False
            col=item.this
            return isinstance(col,exp.Star) if ref['aggregate']=='count' and ref['metric'] is None else isinstance(col,exp.Column) and col.name==ref['metric'] and col.table in ('',src.alias_or_name)
        if isinstance(item,exp.Column) and name in ctes and name not in seen:
            inner=ctes[name]
            found=[v for v in inner.expressions if v.alias_or_name==item.name]
            if len(found)!=1:return False
            return baseline(exp.Subquery(this=inner.copy().select(found[0],append=False)),seen+(name,))
        return False
    def compared(node):
        node=_bare(node);operator=comparison['operator']
        if operator not in OPS:return False
        if isinstance(node,OPS[operator]) and selected_col(node.this,comparison['metric']) and baseline(node.expression):return True
        inverse={'gt':'lt','gte':'lte','lt':'gt','lte':'gte','eq':'eq','neq':'neq'}[operator]
        return isinstance(node,OPS[inverse]) and selected_col(node.expression,comparison['metric']) and baseline(node.this)
    def region_condition(node,alias,region):
        if not region or not isinstance(node,exp.EQ):return False
        for col,val in [(node.this,node.expression),(node.expression,node.this)]:
            if isinstance(col,exp.Column) and col.name=='region_id' and col.table in ('',alias) and isinstance(val,exp.Literal) and val.is_string and val.this==region:return True
        return False
    predicates=_terms((t.args.get('where') or exp.Where()).this)
    region=prepared['regions'][0] if len(prepared['regions'])==1 else None
    predicates=[x for x in predicates if not region_condition(x,primary.alias_or_name,region)]
    projections=[_bare(e) for e in t.expressions]
    if comparison['output'] in {'count','rows'}:
        if len(predicates)!=1 or not compared(predicates[0]):raise PopulationError('WHERE须用对象指标与声明基准比较，不能遗漏、反转或追加筛选。')
        if comparison['output']=='count' and (len(projections)!=1 or not count(projections[0])):raise PopulationError('数量查询须直接COUNT(*)统计比较后的对象。')
        if comparison['output']=='rows' and any(not isinstance(e,exp.Column) or e.table not in ('',primary.alias_or_name) for e in projections):raise PopulationError('样本列表须输出对象集合的真实字段。')
    else:
        if predicates or len(projections)!=1:raise PopulationError('百分比须在未筛掉对象的集合上计算，不能先WHERE过滤分子。')
        expression=projections[0]
        if isinstance(expression,exp.Round):expression=_bare(expression.this)
        if not isinstance(expression,exp.Div):raise PopulationError('百分比必须包含实际除法。')
        numerator,denominator=_bare(expression.this),_bare(expression.expression)
        if not isinstance(numerator,exp.Mul):raise PopulationError('百分比使用100.0乘以分子。')
        a,b=_bare(numerator.this),_bare(numerator.expression)
        def hundred(n):return isinstance(n,exp.Literal) and not n.is_string and n.this=='100.0'
        if hundred(a):numerator=b
        elif hundred(b):numerator=a
        else:raise PopulationError('百分比倍率须为100.0。')
        if not isinstance(denominator,exp.Nullif) or denominator.expression.sql()!='0':raise PopulationError('分母须用NULLIF(...,0)处理空集合。')
        if comparison['operator']=='share':
            valid=count(numerator) and baseline(denominator.this)
        else:
            case=numerator.this if isinstance(numerator,exp.Sum) else None
            cases=case.args.get('ifs',[]) if isinstance(case,exp.Case) else []
            valid=(len(cases)==1 and compared(cases[0].this) and cases[0].args['true'].sql()=='1' and case.args.get('default') is not None and case.args['default'].sql()=='0' and count(denominator.this))
        if not valid:raise PopulationError('百分比分子、分母或基准范围不符合要求。基准必须通过标量子查询读取；声明CTE后不能直接使用未加入FROM的CTE.列名，请改用(SELECT 基准列 FROM 基准CTE)或直接查询reference_samples。')
    original=list(t.find_all(exp.Table))
    for table in original:
        name=table.name.lower()
        if name in RELATIONS:
            alias=table.args.get('alias') or exp.TableAlias(this=exp.to_identifier(table.name))
            relation=prepared['selected' if name=='selected_samples' else 'reference'].copy()
            table.replace(exp.Subquery(this=relation,alias=alias.copy()))
    return t.sql(dialect='sqlite',pretty=True)


def saved_population(prepared,data):
    if prepared['scopes']['selection']['mode']=='previous':
        return copy.deepcopy(prepared['old_population'])
    rows=data.execute(prepared['selector'],max_rows=50)
    if rows['truncated']:raise PopulationError('重选集合超过50张，暂不保存为可指代集合。')
    ids=[str(r['sample_id']) for r in rows['rows']]
    selector=exp.select('sample_id').from_('samples').where(exp.column('sample_id').isin(*[exp.Literal.string(x) for x in ids]) if ids else exp.false())
    return dict(supported=True,kind='displayed_samples',selector_sql=selector.sql(dialect='sqlite'),
                selection_recipe=prepared['recipe'],region_ids=prepared['regions'],
                origin_question=prepared['origin_question'],sample_count=len(ids),
                ordered_ids=ids,filters={'field':'sample_id','operator':'in','values':ids})


def reference_stats(prepared,data):
    """A deterministic availability check, never a replacement query answer."""
    ref=prepared['scopes']['reference']
    metric=exp.column(ref['metric']) if ref['metric'] is not None else exp.Star()
    q=exp.select(AGGS[ref['aggregate']](this=metric.copy()).as_('value'),
                 exp.Count(this=metric.copy()).as_('known'),exp.Count(this=exp.Star()).as_('total')).from_(exp.Subquery(this=prepared['reference'].copy(),alias=exp.TableAlias(this=exp.to_identifier('r'))))
    return data.execute(q.sql(dialect='sqlite'))['rows'][0]


def returned_population(prepared,executed):
    if executed['truncated'] or 'sample_id' not in executed['columns']:
        return dict(supported=False,reason='明细被截断或没有样本ID，不能把可见部分默认为完整集合')
    ids=[str(r['sample_id']) for r in executed['rows']]
    selector=exp.select('sample_id').from_('samples').where(exp.column('sample_id').isin(*[exp.Literal.string(x) for x in ids]) if ids else exp.false())
    return dict(supported=True,kind='displayed_samples',selector_sql=selector.sql(dialect='sqlite'),
                selection_recipe=prepared['recipe'],region_ids=prepared['regions'],
                origin_question='上一轮实际返回的比较结果',sample_count=len(ids),ordered_ids=ids,
                filters={'field':'sample_id','operator':'in','values':ids})


def result_order(prepared):
    tree=sqlglot.parse_one(prepared['recipe'],read='sqlite') if prepared['recipe'] else None
    order=tree.args.get('order') if tree else None
    values=[]
    for item in order.expressions if order else []:
        column=item.this
        if not isinstance(column,exp.Column):raise PopulationError('该选样方法含复杂排序，请明确比较结果的排序。')
        values.append(dict(expression=column.name,direction='DESC' if item.args.get('desc') else 'ASC'))
    return values or [dict(expression='sample_id',direction='ASC')]


def verified_population(population,data,physical_scope):
    p=copy.deepcopy(population)
    q=exp.select('region_id').distinct().from_('samples').where(
        exp.column('sample_id').isin(query=sqlglot.parse_one(p['selector_sql'],read='sqlite')))
    p['region_ids']=sorted(str(r['region_id']) for r in data.execute(q.sql(dialect='sqlite'),region_id=physical_scope)['rows'])
    return p
