"""Model-interpreted query requirements, deterministic references and SQL checks.

These checks constrain an interpretation; they cannot prove the interpretation
matches every possible natural-language question. SQL is always model-authored.
"""
from contextlib import closing
import json
import re
import sqlite3
import time
import uuid

import sqlglot
from sqlglot import exp


class ContractError(ValueError):
    pass


CONTRACT_FIELDS={'action','question','regions','columns','order','limit','offset','per_group','calculations','missing','scopes'}


def planning_context(value):
    """Expose prior requirements without teaching the model internal fields.

    Stored plans also contain program-owned validation annotations. They must
    remain persisted, but are not part of the model's output JSON schema.
    """
    if isinstance(value,list):return [planning_context(item) for item in value]
    if not isinstance(value,dict):return value
    return {key:planning_context({k:v for k,v in item.items() if k in CONTRACT_FIELDS}
            if key in {'contract','queryContract'} and isinstance(item,dict) else
            {k:v for k,v in item.items() if k in {'question','turns','clarification'}} if key=='pending' and isinstance(item,dict) else item)
            for key,item in value.items()}


def named_regions(question, regions):
    found=[]
    for region in regions:
        aliases={region['id'],region.get('short',''),region.get('name','')}
        if any(a and re.search(r'(?<![A-Za-z0-9_])'+re.escape(a)+r'(?![A-Za-z0-9_])',question,re.I) for a in aliases):
            found.append(region['id'])
    return found


def _list(history, first=False):
    if not history:return None
    latest=history.get('last_list')
    item=history.get('root_list') if first else latest
    if item and set(item)=={'list_id'}:return latest if latest and latest['list_id']==item['list_id'] else None
    return item


def _ordinal(text):
    if text.isdigit():return int(text)
    digits={'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    if '十' in text:
        a,b=text.split('十',1);return digits.get(a,1)*10+digits.get(b,0)
    return digits.get(text,0)


def resolve_reference(question,history):
    # Only explicit sample-list references. A fresh rank interval is not a reference.
    match=re.search(r'第([0-9]{1,3}|[一二两三四五六七八九十]{1,3})(?:个|张|条)?',question)
    implicit=bool(match and re.search(r'第[0-9一二两三四五六七八九十]+(?:张|个|条)的',question))
    if not match or not (implicit or re.search(r'刚才|上(?:一轮|面|次)|之前|最开始|最初|一开始|列表中|列表里',question)):return None,None
    if re.search(r'第[0-9一二三四五六七八九十]+(?:到|至|—|-)',question):return None,None
    if re.search(r'第[0-9一二三四五六七八九十]+个区域',question):return None,None
    item=_list(history,bool(re.search(r'最开始|最初|一开始',question)))
    index=_ordinal(match[1]);ids=(item or {}).get('ordered_ids',[])
    if not item or index<1 or index>len(ids):
        return None,'当前会话没有对应的样本列表或该序号超出范围，请先查询列表或提供样本ID。'
    return dict(listId=item['list_id'],position=index,sampleId=ids[index-1]),None


def unbounded_group_summary(question):
    """One aggregate row per region, without any requested row-ranking bound."""
    if not re.search(r'各(?:个)?区域|每(?:个)?区域|按区域(?:分组|汇总|统计)|分区域|所有区域.{0,15}(?:各自|分别)',question):return False
    from query_semantics import regional_total_rank
    if regional_total_rank(question) or re.search(r'哪个|哪一个|哪几个|(?:最高|最低|最多|最少)的区域',question):return False
    if re.search(r'前\s*[0-9一二两三四五六七八九十百]+|第\s*[0-9一二两三四五六七八九十百]+|\btop\s*\d+',question,re.I):return False
    if re.search(r'样本\s*ID|sample_id|逐条|明细|每张|每条|各风险|不同风险|各类别|每种|按.{0,6}(?:风险|等级|类别)|review_level.{0,3}分组',question,re.I):return False
    if re.search(r'(?:最高|最大|最多|最低|最小|最少).{0,8}(?:样本|影像|记录|切片)',question):return False
    return bool(re.search(r'数量|总数|个数|数目|占比|比例|百分比|均值|平均|合计|总和',question))


def normalize_group_summary(contract,question):
    if contract['action']!='query' or not unbounded_group_summary(question):return
    changed=contract['per_group'] or contract['limit'] is not None or contract['offset']!=0
    contract.update(per_group=False,limit=None,offset=0,groupingMode='aggregate')
    if changed:contract.setdefault('normalizations',[]).append('group_summary_not_per_group_topn')


def parse_contract(output,regions,*,source_question=None,schema=None):
    if isinstance(output,dict):
        if output.get('finishReason') in {'length','timeout','cancelled'}:raise ContractError('查询要求未完整生成。')
        text=str(output.get('text') or '')
    else:text=str(output)
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    elif '<think>' in text:raise ContractError('查询要求思考未结束。')
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
    if len(text)>8000:raise ContractError('查询要求过长。')
    try:c=json.loads(text)
    except (ValueError,TypeError) as exc:raise ContractError('查询要求必须是JSON对象。') from exc
    if not isinstance(c,dict) or set(c)-CONTRACT_FIELDS:raise ContractError('查询要求包含未允许的字段。')
    if c.get('action') not in {'query','clarify'}:raise ContractError('action必须是query或clarify。')
    if not isinstance(c.get('question'),str) or not 1<=len(c['question'])<=4000:raise ContractError('必须提供保留全部条件的独立问题。')
    c.setdefault('regions',[]);c.setdefault('columns',[]);c.setdefault('order',[])
    c.setdefault('calculations',[]);c.setdefault('limit',None);c.setdefault('offset',0);c.setdefault('per_group',False);c.setdefault('missing',None)
    if c['order'] is None:c['order']=[]
    if c['offset'] is None and c['limit'] is None:
        c['offset']=0
    if not isinstance(c['regions'],list) or len(c['regions'])>20 or any(not isinstance(r,str) or r not in regions for r in c['regions']):
        raise ContractError('regions必须使用已提供的区域ID。')
    if not isinstance(c['columns'],list) or len(c['columns'])>18 or any(not isinstance(x,str) or not 1<=len(x)<=600 or ';' in x for x in c['columns']):
        raise ContractError('columns必须是输出列名、别名或单个计算表达式的列表。')
    for column in c['columns']:
        try:parsed=sqlglot.parse_one(column,read='sqlite')
        except sqlglot.errors.ParseError as exc:raise ContractError('输出列无法解析。') from exc
        if isinstance(parsed,exp.Query) or parsed.find(exp.Subquery) or isinstance(parsed,exp.Star) or any(isinstance(s.parent,exp.Column) for s in parsed.find_all(exp.Star)):
            raise ContractError('输出列不能是查询或裸通配符；COUNT(*)计算表达式可以使用。')
    if len(set(c['columns']))!=len(c['columns']):raise ContractError('输出列名不能重复。')
    if not isinstance(c['order'],list) or len(c['order'])>8:raise ContractError('排序要求无效。')
    from model_sql import SCHEMA as DEFAULT_SCHEMA
    SCHEMA = schema or DEFAULT_SCHEMA
    available_columns=set(re.findall(r'\b([a-z_]+)\s+(?:TEXT|INTEGER|REAL)\b',SCHEMA))
    output_aliases={p.alias for p in (sqlglot.parse_one(v,read='sqlite') for v in c['columns']) if isinstance(p,exp.Alias)}
    for item in c['order']:
        if not isinstance(item,dict) or set(item)!={'expression','direction'} or item['direction'] not in {'ASC','DESC'}:
            raise ContractError('order必须包含expression和ASC/DESC方向。')
        if not isinstance(item['expression'],str) or not 1<=len(item['expression'])<=600 or ';' in item['expression']:
            raise ContractError('排序表达式无效。')
        try:parsed=sqlglot.parse_one(item['expression'],read='sqlite')
        except sqlglot.errors.ParseError as exc:raise ContractError('排序表达式无法解析。') from exc
        if isinstance(parsed,exp.Query) or parsed.find(exp.Subquery):raise ContractError('排序要求不能包含独立查询。')
        if isinstance(parsed,exp.Column) and parsed.name not in available_columns and parsed.name not in c['columns'] and parsed.name not in output_aliases:
            raise ContractError('排序使用未定义的别名'+parsed.name+'；请使用已声明的输出别名、真实字段或完整聚合表达式。')
    for key,maximum in [('limit',50),('offset',10000)]:
        v=c[key]
        if key=='limit' and v is None:continue
        if type(v) is not int or v<0 or v>maximum or (key=='limit' and v==0):raise ContractError(key+'超出范围。')
    if type(c['per_group']) is not bool:raise ContractError('per_group必须是布尔值。')
    if source_question is not None:normalize_group_summary(c,source_question)
    if c['per_group'] and (c['limit'] is None or c['offset']):raise ContractError('每组排名必须提供limit，目前只支持每组前N条、offset=0。')
    # Different instruction models express scoped calculations as records or prose.
    # Preserve the content instead of rejecting a usable plan or deleting its scope.
    calculations=c['calculations']
    if calculations is None:calculations=[]
    if isinstance(calculations,dict):calculations=[calculations]
    if not isinstance(calculations,list) or len(calculations)>8 or any(not isinstance(x,(str,dict)) for x in calculations):
        raise ContractError('计算要求无效。')
    c['calculations']=[json.dumps(x,ensure_ascii=False,separators=(',',':')) if isinstance(x,dict) else x for x in calculations]
    if any(len(x)>1600 for x in c['calculations']):raise ContractError('单项计算要求过长。')
    if c.get('scopes') is not None:
        from query_scopes import parse_scopes
        c['scopes']=parse_scopes(c['scopes'],regions)
    if c['action']=='query':
        if not c['columns'] or c['missing']:raise ContractError('可查询请求必须声明输出列，missing为null。')
    else:
        m=c['missing']
        if m is None and c['columns']:
            # Normalize an internally contradictory enum, not a missing reason.
            # SQL generation independently retains the right to report missing data.
            c['action']='query'
            c['normalizations']=['action_from_complete_query']
            return c
        if m is None:raise ContractError('action=clarify却没有缺失条件。条件完整时请改为action=query；只有确实缺失字段/选择时才填写missing并澄清。')
        if not isinstance(m,dict) or set(m)!={'kind','field','question'} or m['kind'] not in {'unavailable_field','user_choice'}:
            raise ContractError('只能澄清真实缺失字段或用户未给出的选择，均值等计算值应通过SQL求得。')
        if not all(isinstance(m[k],str) and 1<=len(m[k])<=800 for k in ('field','question')):raise ContractError('必须具体说明缺失条件。')
        if m['kind']=='user_choice' and m['field'] not in {'sort_metric','threshold','region','time_window','business_definition'}:
            raise ContractError('用户选择字段无效；不要让用户提供数据库均值、数量或占比。')
        if m['kind']=='unavailable_field' and re.search(r'avg|mean|average|sum|count|平均|均值|总数|占比',m['field'],re.I):
            raise ContractError('聚合中间值可由数据库计算，不能要求用户提供。')
        if m['kind']=='unavailable_field':
            from query_semantics import derived_risk_field
            if 'review_level' in SCHEMA and derived_risk_field(m['field']):raise ContractError('风险数量和占比可由review_level条件聚合计算，输出别名不是缺失的原始字段。')
            available=set(re.findall(r'\b([a-z_]+)\s+(?:TEXT|INTEGER|REAL)\b',SCHEMA))
            if m['field'].strip('`" ').lower() in available:raise ContractError('该字段已在数据库中，不能要求用户再次提供。')
    return c


def _canonical(expression,query):
    tree=expression.copy()
    primary=(query.args.get('from_') or exp.From()).this
    sources=[primary]+[j.this for j in query.args.get('joins') or []]
    preserve=len(sources)>1
    aliases={s.alias_or_name.lower():('_primary' if i==0 else '_joined_'+str(i)) for i,s in enumerate(sources) if s is not None}
    equivalent={}
    def source_key(column):
        return aliases.get(column.table.lower(),column.table.lower()) if column.table else '_primary'
    def representative(key):
        while key in equivalent:key=equivalent[key]
        return key
    # Only mandatory INNER equijoin keys are interchangeable. Never erase all
    # qualifiers: other columns of joined/self-joined rows remain different.
    for join in query.args.get('joins') or []:
        if join.args.get('side') or str(join.args.get('kind') or '').upper() not in ('','INNER'):continue
        for term in _conjuncts(join.args.get('on')):
            if isinstance(term,exp.EQ) and isinstance(term.this,exp.Column) and isinstance(term.expression,exp.Column) and term.this.name==term.expression.name=='region_id':
                a=representative(source_key(term.this));b=representative(source_key(term.expression))
                if a!=b:equivalent[b]=a
    for col in tree.find_all(exp.Column):
        # Distinct joined sources stay distinct, including CTEs and case variants.
        if preserve:
            name=source_key(col)
            if col.name=='region_id':name=representative(name)
            col.set('table',exp.to_identifier(name))
        else:col.set('table',None)
        col.set('db',None);col.set('catalog',None)
    for identifier in tree.find_all(exp.Identifier):identifier.set('this',identifier.this.lower())
    return tree.sql(dialect='sqlite')


def _conjuncts(expression):
    if isinstance(expression,exp.Paren):return _conjuncts(expression.this)
    if isinstance(expression,exp.And):return _conjuncts(expression.this)+_conjuncts(expression.expression)
    return [expression] if expression is not None else []


def _ranked_source(query):
    """Recognize the supported direct CTE/subquery ROW_NUMBER top-N shape."""
    source=(query.args.get('from_') or exp.From()).this
    if isinstance(source,exp.Subquery):return source.this,source.alias_or_name
    if isinstance(source,exp.Table):
        for cte in query.ctes:
            if cte.alias==source.name:return cte.this,source.alias_or_name
    return None,None


def _rank_constraints(query,names,qualifier):
    for term in _conjuncts((query.args.get('where') or exp.Where()).this):
        if not any(col.name in names and col.table in ('',qualifier) for col in term.find_all(exp.Column)):continue
        col=term.this
        if not isinstance(col,exp.Column) or col.name not in names or col.table not in ('',qualifier):
            raise ContractError('排名条件必须是明确的AND区间约束，不能额外隐藏在OR或表达式中。')
        yield term


def _validate_group_limit(tree,c):
    if tree.args.get('limit') or tree.args.get('offset') or tree.args.get('joins'):
        raise ContractError('每组排名须直接查询排名子查询，不能使用全局分页或外层连接。')
    source,qualifier=_ranked_source(tree)
    if not isinstance(source,exp.Select):raise ContractError('每组前N条需要按region_id分区的ROW_NUMBER及排名上界。')
    base=(source.args.get('from_') or exp.From()).this
    if not isinstance(base,exp.Table) or base.name.lower()!='samples' or source.args.get('joins') or source.ctes or source.args.get('limit') or source.args.get('offset'):
        raise ContractError('每组排名目前只支持samples单一来源的直接排名，不能混用其他来源的分区。')
    ranks=[]
    for expression in source.expressions:
        if not isinstance(expression,exp.Alias) or not isinstance(expression.this,exp.Window):continue
        window=expression.this
        partition=window.args.get('partition_by') or []
        if isinstance(window.this,exp.RowNumber) and len(partition)==1 and isinstance(partition[0],exp.Column) and partition[0].name=='region_id' and partition[0].table in ('',base.alias_or_name):
            wanted=c.get('rankOrder')
            if wanted:
                actual=list((window.args.get('order') or exp.Order()).expressions)
                if len(actual)<len(wanted):raise ContractError('每组窗口排名缺少要求的选样排序。')
                _check_order(actual[:len(wanted)],wanted,source)
                extra=actual[len(wanted):]
                if extra and (len(extra)!=1 or not isinstance(extra[0].this,exp.Column) or extra[0].this.name!='sample_id' or extra[0].args.get('desc')):
                    raise ContractError('窗口排名只能在指定排序后追加sample_id升序作为稳定同值处理。')
            ranks.append(expression.alias)
    bounded=False
    for term in _rank_constraints(tree,ranks,qualifier):
        if isinstance(term,exp.LTE) and term.expression.sql()==str(c['limit']):bounded=True
        elif isinstance(term,exp.EQ) and c['limit']==1 and term.expression.sql()=='1':bounded=True
        elif isinstance(term,exp.Between) and term.args['low'].sql()=='1' and term.args['high'].sql()==str(c['limit']):bounded=True
        elif isinstance(term,exp.GTE) and term.expression.sql()=='1' or isinstance(term,exp.GT) and term.expression.sql()=='0':continue
        else:raise ContractError('每组前N条包含额外或冲突的排名约束。')
    if bounded:return
    raise ContractError('每组前N条缺少强制的排名上界（rn<=N），不能返回全部样本。')


def _validate_reference(tree,reference):
    source=(tree.args.get('from_') or exp.From()).this
    if not isinstance(tree,exp.Select) or not isinstance(source,exp.Table) or source.name.lower()!='samples' or tree.ctes:
        raise ContractError('样本引用必须直接从samples查询已绑定ID，不能通过其他来源或常量伪造ID。')
    joins=tree.args.get('joins') or []
    if any(not isinstance(j.this,exp.Table) or j.this.name!='regions' for j in joins):
        raise ContractError('样本引用不支持连接其他样本行。')
    terms=_conjuncts((tree.args.get('where') or exp.Where()).this)
    for term in terms:
        if not isinstance(term,exp.EQ):continue
        for column,literal in [(term.this,term.expression),(term.expression,term.this)]:
            if isinstance(column,exp.Column) and column.name=='sample_id' and column.table in ('',source.alias_or_name) and isinstance(literal,exp.Literal) and literal.is_string and literal.this==reference['sampleId']:
                return
    raise ContractError('SQL必须以AND条件直接绑定samples.sample_id为上一轮的真实ID。')


def _check_order(actual,wanted,query,non_null_sample_count=False):
    if len(actual)!=len(wanted):raise ContractError('外层ORDER BY缺失或排序字段数量不符。')
    sample_aliases={t.alias_or_name.lower() for t in query.find_all(exp.Table) if t.name.lower()=='samples'}
    def canonical(expression):
        if non_null_sample_count:
            expression=expression.copy()
            for count in expression.find_all(exp.Count):
                column=count.this
                if isinstance(column,exp.Column) and column.name=='sample_id' and (not column.table or column.table.lower() in sample_aliases):count.set('this',exp.Star())
        return _canonical(expression,query)
    aliases={canonical(exp.column(e.alias)):canonical(e.this) for e in query.expressions if isinstance(e,exp.Alias)}
    for a,w in zip(actual,wanted):
        left=canonical(a.this);right=canonical(sqlglot.parse_one(w['expression'],read='sqlite'))
        if aliases.get(left,left)!=aliases.get(right,right) or bool(a.args.get('desc'))!=(w['direction']=='DESC'):
            raise ContractError('外层排序字段或方向不符合查询要求。')


def _validate_window_page(tree,c):
    """Accept ONE ranked page, not a rank filter combined with another LIMIT.

    The rank must be directly exported by the source, ordered like the final
    contract, and bounded to exactly the requested interval. Do not rewrite SQL.
    """
    source,qualifier=_ranked_source(tree)
    windows=list(tree.find_all(exp.Window))
    if tree.args.get('limit') or tree.args.get('offset') or tree.args.get('joins') or not isinstance(source,exp.Select) or source.args.get('joins') or len(windows)!=1:
        raise ContractError('窗口分页须使用单一排名来源，不能与外层分页/连接重复组合。')
    window=windows[0]
    if not isinstance(window.this,exp.RowNumber) or window.args.get('partition_by') or not isinstance(window.parent,exp.Alias) or window.parent not in source.expressions:
        raise ContractError('全局窗口分页必须是直接输出的ROW_NUMBER，不能分组或通过重命名来源隐藏排名。')
    if any(s.args.get('limit') or s.args.get('offset') for s in tree.find_all(exp.Select)):
        raise ContractError('窗口分页不能混用内层LIMIT/OFFSET。')
    _check_order(list(window.args['order'].expressions) if window.args.get('order') else [],c['order'],source)
    rank=window.parent.alias;lo=c['offset']+1;hi=c['offset']+c['limit']
    low_ok=c['offset']==0;high_ok=False
    for term in _rank_constraints(tree,{rank},qualifier):
        if isinstance(term,exp.Between) and term.args['low'].sql()==str(lo) and term.args['high'].sql()==str(hi):low_ok=high_ok=True
        elif isinstance(term,exp.EQ) and lo==hi and term.expression.sql()==str(lo):low_ok=high_ok=True
        elif isinstance(term,exp.LTE) and term.expression.sql()==str(hi):high_ok=True
        elif isinstance(term,exp.GTE) and term.expression.sql()==str(lo) or isinstance(term,exp.GT) and term.expression.sql()==str(lo-1):low_ok=True
        else:raise ContractError('窗口分页包含额外或冲突的排名约束。')
    if not (low_ok and high_ok):raise ContractError('窗口排名边界不符合所需区间：'+str(lo)+'—'+str(hi)+'。')


def _count_scope_only(tree,regions):
    """Allow only NULL exclusion and region predicates covering the full scope."""
    def allowed(node):
        if isinstance(node,exp.Paren):return allowed(node.this)
        if isinstance(node,exp.And):return allowed(node.this) and allowed(node.expression)
        if isinstance(node,exp.Not) and isinstance(node.this,exp.Is):
            return isinstance(node.this.this,exp.Column) and node.this.this.name=='estimated_building_count' and isinstance(node.this.expression,exp.Null)
        if isinstance(node,exp.EQ):
            a,b=node.this,node.expression
            if isinstance(a,exp.Literal):a,b=b,a
            return isinstance(a,exp.Column) and a.name=='region_id' and isinstance(b,exp.Literal) and b.is_string and bool(regions) and set(regions)<={b.this}
        if isinstance(node,exp.In) and not node.args.get('query'):
            return isinstance(node.this,exp.Column) and node.this.name=='region_id' and all(isinstance(v,exp.Literal) and v.is_string for v in node.expressions) and bool(regions) and set(regions)<={v.this for v in node.expressions}
        return False
    return all(allowed(w.this) for w in tree.find_all(exp.Where))


def validate_sql(sql,c,scope=None,reference=None,*,total_sample_rank=False):
    tree=sqlglot.parse_one(sql,read='sqlite')
    if c.get('buildingTotal'):
        projection=tree.expressions[0] if isinstance(tree,exp.Select) and len(tree.expressions)==1 else None
        if isinstance(projection,exp.Alias):projection=projection.this
        if (not isinstance(projection,exp.Sum) or not isinstance(projection.this,exp.Column) or projection.this.name!='estimated_building_count'
            or tree.ctes or tree.find(exp.Subquery) or any(tree.args.get(k) for k in ('group','having','joins','order','limit','offset')) or not _count_scope_only(tree,c.get('buildingCountRegions',[]))):
            raise ContractError('本轮是当前批次建筑量总量，只返回SUM(estimated_building_count)，不返回图片列表、不排名、不分组、不筛选。')
    if c.get('requiresBuildingEstimate') and not any(n.name.lower()=='estimated_building_count' for n in tree.find_all(exp.Column)):
        raise ContractError('建筑量必须使用estimated_building_count；图片行数COUNT(*)和未过滤component_count不能代替建筑量。')
    if c.get('unfilteredBuildingRank'):
        if not _count_scope_only(tree,c.get('buildingCountRegions',[])):
            raise ContractError('本轮要求本批全部图片按建筑量排名，不能添加已复核、未复核或其他筛选。移除整个WHERE条件，包括estimated_building_count > 0；建筑量为0的图片也必须保留。只需要直接FROM samples ORDER BY所需指标 LIMIT/OFFSET。')
        if tree.find(exp.Having):raise ContractError('本轮建筑量排名没有要求聚合后筛选，不能添加HAVING。')
    if c.get('simpleSampleRank') and (tree.ctes or tree.find(exp.Subquery) or tree.find(exp.Window)):
        raise ContractError('本轮只是按单张指标重新排名：直接保留原筛选并ORDER BY、LIMIT 1 OFFSET N-1，不增加MAX子查询或窗口再排一次。')
    if c.get('includeZeroForeground'):
        averages=list(tree.find_all(exp.Avg))
        if (not averages or any(not isinstance(a.this,exp.Column) or a.this.name!='foreground_ratio' for a in averages)
            or tree.ctes or tree.find(exp.Subquery) or any(isinstance(n,(exp.Where,exp.Having,exp.Join,exp.Filter)) for n in tree.walk())):
            raise ContractError('本批全部影像的建筑占比均值直接AVG(foreground_ratio)，必须包含0；本轮没有筛选，不加WHERE、NULLIF、CASE或子查询。')
    if c.get('groupingMode')=='aggregate':
        if tree.args.get('limit') or tree.args.get('offset') or tree.find(exp.Window):
            raise ContractError('本轮是各区域一行的汇总，没有每组前N名或分页要求；不要使用窗口排名或LIMIT/OFFSET。')
        group=(tree.args.get('group') or exp.Group()).expressions
        if not group or any(not isinstance(g,exp.Column) or g.name not in {'region_id','region_name'} for g in group):
            raise ContractError('本轮须按区域汇总，每个区域一行；请直接GROUP BY region_id，可同时保留region_name，不要按样本或风险等级拆组。')
    columns=[e.alias_or_name for e in tree.expressions]
    def matches_column(actual,wanted):
        if actual.alias_or_name==wanted:return True
        parsed=sqlglot.parse_one(wanted,read='sqlite')
        if isinstance(parsed,exp.Alias) and actual.alias_or_name!=parsed.alias:return False
        return _canonical(actual.this if isinstance(actual,exp.Alias) else actual,tree)==_canonical(parsed.this if isinstance(parsed,exp.Alias) else parsed,tree)
    matches=len(columns)==len(c['columns']) and all(matches_column(a,w) for a,w in zip(tree.expressions,c['columns']))
    if not matches:raise ContractError('输出列不符合查询要求：应为'+','.join(c['columns'])+'，实际为'+','.join(columns))
    actual=list(tree.args.get('order').expressions) if tree.args.get('order') else []
    if c.get('orderRequired',True):_check_order(actual,c['order'],tree,non_null_sample_count=total_sample_rank)
    if c['limit'] is not None and not c['per_group']:
        limit=tree.args.get('limit');offset=tree.args.get('offset')
        if not limit and not offset:
            _validate_window_page(tree,c)
        elif not limit or limit.expression.sql()!=str(c['limit']) or (offset.expression.sql() if offset else '0')!=str(c['offset']):
            raise ContractError('全局分页不符合要求，请在外层使用LIMIT '+str(c['limit'])+' OFFSET '+str(c['offset'])+'，不要重复使用窗口分页。')
        elif any(s is not tree and (s.args.get('limit') or s.args.get('offset')) for s in tree.find_all(exp.Select)) or any(isinstance(w.this,(exp.RowNumber,exp.Rank,exp.DenseRank)) for w in tree.find_all(exp.Window)):
            raise ContractError('全局分页不能混用内层分页或排名窗口；请只在最外层LIMIT/OFFSET一次分页。')
    elif not c['per_group'] and (tree.args.get('limit') or tree.args.get('offset')):
        raise ContractError('用户没有要求截断或分页，不能擅自添加外层LIMIT/OFFSET。')
    if c['per_group']:_validate_group_limit(tree,c)
    if reference:_validate_reference(tree,reference)
    return sql


def remember(history,q,c,executed,population=None):
    from query_dialogue import frame_from_sql
    previous=history or {}
    latest=_list(previous);first=_list(previous,True)
    ids=[r['sample_id'] for r in executed['rows'] if r.get('sample_id')]
    if ids:
        latest=dict(list_id=uuid.uuid4().hex,question=q,contract=c,ordered_ids=ids)
        if first is None:first=latest
    return dict(question=q,sql=executed['sql'],columns=executed['columns'],
        rows=executed['rows'][:5] if ids and len(executed['rows'])>10 else executed['rows'],
        rowsPreviewTruncated=bool(ids and len(executed['rows'])>10),truncated=executed['truncated'],
        queryContract=c,last_list=latest,population=population,taskFrame=frame_from_sql(executed['sql'],c['question']),
        root_list={'list_id':latest['list_id']} if first and latest and first['list_id']==latest['list_id'] else first)


def run_contract(service,q,scope,owner,data,history,result,deadline,started,*,selection=None):
    from model_sql import bounded_context,parse_output,read_sql,review_version,Snapshot,StaleData
    from query_state import resolve_population, derive_population, bind_population, validate_analysis, analysis_state, reference_population, PopulationError
    from query_scopes import requested as scoped_request, prepare as prepare_scopes, validate_and_expand, saved_population, returned_population, result_order, reference_stats, interpretation_prompt, compile_contract, verified_population
    import query_clarification as decisions
    current_context=history
    input_question=q
    building_total=service.task_type=='building_segmentation' and bool(re.fullmatch(r'(?:请)?(?:统计|查询|计算)?(?:本批|这批|全批)(?:全部|所有)?(?:图片|影像)?(?:的)?(?:建筑量(?:合计|总量)?(?:是多少)?|有多少栋建筑|有几栋建筑|建筑(?:物)?(?:数量|总数)(?:是多少)?)[。？?！!]*',q.strip()))
    if building_total:
        # A complete whole-batch total supersedes a prior image/ranking task.
        # Do not feed unrelated list/rank state back into interpretation.
        history={};current_context=history
    selected_binding=None
    def finish():
        pending=decisions.public_pending(current_context)
        if pending:result['clarification']=pending
        if (current_context or {}).get('semanticState'):result['queryState']=current_context['semanticState']
        result['latencyMs']=round((time.monotonic()-started)*1000,2)
        result['toolTrace']=[dict(tool='model_sql',status=result['status'],summary=f'查询要求→SQL校验→只读执行；实际模型调用{result["llmCalls"]}次')]
        return result
    def clarify(message,source):
        result.update(status='needs_clarification',answer=message,clarificationSource=source)
        if not decisions.public_pending(current_context):result['clarification']=decisions.problem(message,source)
        return finish()
    def invoke(prompt,stage,tokens):
        if len(prompt)>16000:raise ContractError('上下文超过长度预算，请开始新会话。')
        if time.monotonic()>=deadline:raise ContractError('生成预算已用尽。')
        result['llmCalls']+=1
        extra={'use_query_adapter':True} if stage=='state_update' and service.query_update_adapter else {}
        output=service.generate(prompt,max_tokens=tokens,deadline=deadline,**extra)
        if isinstance(output,dict):result['modelUsage'].append(dict(stage=stage,**{k:output[k] for k in ('inputTokens','outputTokens','finishReason','latencyMs') if k in output}))
        if review_version(service.reviewed_provider())!=data.review_version or Snapshot(service.source_path).source_version!=data.version:
            raise StaleData('生成期间数据或复核记录更新，请重新提问。')
        return output
    def save(context):
        nonlocal current_context
        semantic=decisions.semantic_state(context,data.source.regions)
        if semantic:context['semanticState']={**semantic,'dataVersion':data.data_version}
        bounded_context(context)
        with closing(sqlite3.connect(service.history_path)) as db,db:
            db.execute('INSERT OR REPLACE INTO query_history VALUES(?,?,?,?)',(owner,data.data_version,scope or '',json.dumps(context,ensure_ascii=False)))
        current_context=context
    def save_pending(message,metric_choice=False,missing=None):
        metadata=missing or dict(kind='user_choice',field='business_definition')
        pending=decisions.model_pending(history,q,message,metadata,data.data_version,data.source.regions,scope)
        save({**(history or {}),'pending':pending})
    def save_scope_attempt():
        if multi_scope:save({**(history or {}),'scope_pending_question':q})

    reply=decisions.resolve_reply(history,selection or {},q,data.data_version)
    if reply and reply.get('error'):return clarify(reply['error'],reply['source'])
    if reply:
        original_q=q;q=reply['question'];selected_binding=reply['binding']
        result['clarificationResolution']=reply['clarificationResolution']
        if selected_binding['type']=='metric':result['metricResolution']=dict(originalQuestion=original_q,resolvedQuestion=q,measure=selected_binding['definition'])
    elif decisions.public_pending(history):
        # A new query supersedes the prior question before any model call.
        # A stale button may never revive that prior task, even if this query fails.
        history={k:v for k,v in history.items() if k!='pending'}
        save(history)
    if service.task_type == 'building_segmentation':
        from task_profiles import segmentation_decision
        message = None if selected_binding and selected_binding.get('type')=='segmentation_metric' else segmentation_decision(q)
        if message:
            options=[dict(label='统计建筑连通区域数',value=dict(type='segmentation_metric',question='建筑掩膜连通区域总数是多少？')),
                     dict(label='统计含建筑图片数',value=dict(type='segmentation_metric',question='含建筑的图片有多少张？'))]
            pending=decisions.make_pending(history,q,message,'segmentation_measure',options,data.data_version,data.source.regions)
            save({**(history or {}),'pending':pending})
            return clarify(message,'segmentation_measure')
    capture_time=bool(re.search(r'20\d{2}年|去年|今年|(?:拍摄|采集|获取|观测)(?:的)?(?:日期|时间|年份)',q))
    cancelled_time=bool(re.search(r'不按.{0,6}(?:年份|日期|时间)|不(?:限制|限定).{0,8}(?:年份|日期|时间)|(?:忽略|取消|去掉).{0,8}(?:年份|日期|时间|年)',q))
    if capture_time and not cancelled_time:
        message='当前样本快照没有拍摄/采集日期字段，无法按指定年份或时间筛选。可以取消时间条件后继续统计；不会忽略年份直接返回结果。'
        save_pending(message,missing=dict(kind='unavailable_field',field='capture_date'))
        return clarify(message,'unavailable_field_guard')

    from query_updates import run_update,run_sample_rank,run_segmentation_measure_switch
    switched=run_segmentation_measure_switch(service,q,scope,data,history,result,save,finish) if not reply else None
    if switched is not None:return switched
    sample_rank=run_sample_rank(service,q,scope,data,history,result,save,finish) if not reply else None
    if sample_rank is not None:return sample_rank
    updated=run_update(service,q,scope,data,history,result,invoke,save,finish) if not reply else None
    if updated is not None:return updated

    pending_decision=decisions.detect(history,q,data.source.regions,scope,data.data_version) if not reply else None
    if pending_decision:
        save({**(history or {}),'pending':pending_decision})
        source='metric_definition' if pending_decision['decision']['slot']=='measure_definition' else 'region_reference_guard' if pending_decision['decision']['slot']=='region' else 'query_decision'
        return clarify(pending_decision['clarification'],source)

    from query_comparisons import run_comparison
    compared=run_comparison(service,q,scope,data,history,result,invoke,save,finish,clarify,(selected_binding or {}).get('measure') if (selected_binding or {}).get('type')=='comparison_measure' else None)
    if compared is not None:return compared

    from query_dialogue import context_request,resolution_prompt,parse_resolution,validate_question,validate_measures,validate_measure_order,label,ambiguous_hole_metric,hole_choice,HOLE_CHOICES,explicit_measures,metric_change,task_frame,measure_family
    metric_pending=(history or {}).get('pending') or {}
    choice=hole_choice(q)
    if not reply and metric_pending.get('metricChoice') and choice is not None:
        original_q=q
        q=metric_pending['question'].rstrip('。？?')+'；明确统计口径：'+HOLE_CHOICES[choice]+'。'
        result['metricResolution']=dict(originalQuestion=original_q,resolvedQuestion=q,measure=HOLE_CHOICES[choice])
    elif not reply and ambiguous_hole_metric(q):
        previous_measures=(task_frame(history) or {}).get('measures',[])
        previous_holes=[m for m in previous_measures if measure_family(m)=='holes']
        known_choice=0 if any(m['operation']=='flag_percentage' for m in previous_holes) else 1 if any(m['field']=='hole_ratio' and m['operation']=='avg' for m in previous_holes) else None
        if known_choice is not None and len(previous_holes)==len(previous_measures):
            original_q=q;q=q.rstrip('。？?')+'；沿用已明确的统计口径：'+HOLE_CHOICES[known_choice]+'。'
            result['metricResolution']=dict(originalQuestion=original_q,resolvedQuestion=q,measure=HOLE_CHOICES[known_choice],source='previous_query_definition')
        else:
            # Catalog decisions above own new questions. This fallback only
            # migrates an old, pre-protocol pending state.
            pending=decisions.make_pending(history,q,'空洞比例要按哪种口径计算？','measure_definition',decisions.metric_options('holes'),data.data_version,data.source.regions,family='holes')
            save({**(history or {}),'pending':pending})
            return clarify(pending['clarification'],'metric_definition')
    region_reference=decisions.unresolved_region_reference(q,data.source.regions)
    if region_reference:
        aliases={name.lower():region['id'] for region in data.source.regions for name in (region['id'],region.get('short',''),region.get('name','')) if name}
        rows=(history or {}).get('rows') or []
        ids={aliases[str(row[key]).lower()] for row in rows for key in ('region_id','region_name') if str(row.get(key,'')).lower() in aliases}
        if len(ids)!=1 or (history or {}).get('truncated') or (history or {}).get('rowsPreviewTruncated'):
            return clarify('上一轮结果没有唯一确定一个区域，请说明要继续查看哪个区域。','region_reference_guard')
        region_id=next(iter(ids));original_q=q
        q=q[:region_reference.start()]+region_id.upper()+'区域'+q[region_reference.end():]
        result['resolvedRegionReference']=dict(regionId=region_id,source='previous_executed_rows',originalQuestion=original_q,resolvedQuestion=q)
    declared_regions=named_regions(q,data.source.regions)
    global_region_request=bool(re.search(r'各(?:个)?区域|每(?:个)?区域|所有区域|全部区域|(?:对比|比较).{0,8}(?:其他|其它|别的)区域',q))
    changed_metric=metric_change(q,history)
    if changed_metric:
        result['metricUpdate']=dict(previousMeasures=[label(m) for m in changed_metric['previous']],requestedMeasures=[label(m) for m in changed_metric['requested']],families=changed_metric['families'])
        from query_state import DEICTIC
        if not declared_regions and not global_region_request and not DEICTIC.search(q):
            # A new projection does not erase an explicitly established region.
            previous_regions=list(((history or {}).get('queryContract') or {}).get('regions') or [])
            if previous_regions:
                q+='本轮沿用区域：'+ '、'.join(previous_regions)+'。'
                declared_regions=previous_regions
    dialogue=None if changed_metric else context_request(q,history)
    if dialogue:
        dialogue['currentRegions']=declared_regions
        if selected_binding and selected_binding['type']=='sort':dialogue['sort_measure']=selected_binding['measure']
        resolved=None;context_error=''
        for attempt in range(2):
            context_preview=''
            try:
                output=invoke(resolution_prompt(dialogue)+('\n上次校验错误='+context_error if context_error else ''),'context',768)
                context_preview=str(output.get('text','') if isinstance(output,dict) else output)[:5000]
                resolved=parse_resolution(output,dialogue)
                result['attempts'].append(dict(stage='context',status='clarification' if resolved['mode']=='clarify' else 'resolved',resolution=resolved))
                break
            except Exception as exc:
                context_error=str(exc)[:400];result['attempts'].append(dict(stage='context',status='rejected',error=context_error,resolutionPreview=context_preview))
                if isinstance(exc,StaleData):break
        if resolved is None:
            result.update(status='failed',answer='未能可靠补全上一轮统计目标，未执行SQL。',error=context_error);return finish()
        if resolved['mode']=='clarify':
            save_pending(resolved['clarification'])
            return clarify(resolved['clarification'],'dialogue_resolution')
        result['contextResolution']=dict(originalQuestion=q,previousQuestion=dialogue['frame']['question'],
            resolvedQuestion=resolved['question'],modelResolvedQuestion=resolved.get('modelQuestion'),programBoundMeasures=resolved.get('programBoundMeasures',False),
            programBoundSelection=resolved.get('programBoundSelection',False),
            mode=resolved['mode'],retainedMeasures=[label(m) for m in dialogue['required']])
        q=resolved['question']

    explicit=named_regions(q,data.source.regions)
    comparing_regions=bool(dialogue and dialogue.get('compare_regions'))
    if declared_regions and dialogue and not comparing_regions:
        explicit=declared_regions
        q+='本轮查询范围：'+'、'.join(declared_regions)+'。'
    inherited_candidates=None
    from query_semantics import regional_total_rank
    scope_preserving_edit=bool(dialogue and not declared_regions and not comparing_regions and not global_region_request and not regional_total_rank(dialogue['original']))
    if dialogue and not declared_regions and (scope_preserving_edit or dialogue.get('sort_direction') or dialogue.get('extreme_direction') or global_region_request):
        inherited_candidates=list(((history or {}).get('queryContract') or {}).get('regions') or [])
        if global_region_request:inherited_candidates=[]
        explicit=inherited_candidates
        q+='本轮候选区域：'+('全部区域' if not inherited_candidates else '、'.join(inherited_candidates))+'。'
    if comparing_regions:
        if scope:return clarify('当前只选择了'+scope.upper()+'，无法与其他区域比较；请切换到全部区域后重新提问。','scope_guard')
        # The previous named region is the comparison anchor, not a WHERE
        # restriction on the new all-region report.
        explicit=[]
    if dialogue:result['contextResolution']['resolvedQuestion']=q
    if scope and re.search(r'全库|全部区域|所有区域',q):
        return clarify('当前只选择了'+scope.upper()+'，无法计算全库或全部区域口径；请切换到全部区域后重新提问。','scope_guard')
    if scope and any(r!=scope for r in explicit):
        return clarify('当前只选择了'+scope.upper()+'，无法完整查询'+ '、'.join(r.upper() for r in explicit)+'；请切换到全部区域后重新提问。','scope_guard')
    reference,message=resolve_reference(q,history)
    if service.task_type=='building_segmentation' and not reference:
        matched=re.search(r'样本\s*ID(?:为|是|=|：|:)?\s*([A-Za-z0-9_.-]+)',q,re.I)
        if matched:
            sid=matched[1]
            matches=[r for region in data.source.regions for r in data.source.rows(region['id']) if r['sample_id']==sid and (not scope or r['region_id']==scope)]
            if not matches:return clarify('当前批次和查询范围内没有该图片，请重新选择。','sample_reference')
            reference=dict(sampleId=sid,source='verified_sample_id');message=None
    if message:return clarify(message,'reference_guard')
    if reference:result['resolvedReference']=reference
    multi_scope=scoped_request(q,history) and not reference
    if multi_scope and not ((history or {}).get('population') or {}).get('supported'):
        return clarify('当前会话没有可绑定的样本集合，请先查询一批样本；数据变更后须重新选样。','population_guard')
    try:population=(history or {}).get('population') if multi_scope else None if reference else resolve_population(q,history)
    except PopulationError as exc:return clarify(str(exc),'population_guard')
    if multi_scope:
        population=verified_population(population,data,scope)
        history={**history,'population':population}
    if not multi_scope and population and population.get('region_ids') and any(r not in population['region_ids'] for r in explicit):
        return clarify('本轮明确提及原样本集合之外的区域，请写出完整的新查询条件；不会静默忽略新增区域。','population_guard')
    if population:result['population']={**population,'inherited':True}
    first=_list(history,True)
    if reference:
        referred=_list(history,bool(re.search(r'最开始|最初|一开始',q)))
        context=dict(resolvedReference=reference)
        if re.search(r'列表|这批|那批|这些|平均|均值',q) and referred:
            context['referencedList']=dict(question=referred['question'],contract=referred['contract'])
    elif re.search(r'回到|最开始|最初|一开始',q) and first:
        context=dict(originalList=dict(question=first['question'],contract=first['contract']))
    else:context=dict(previousTask=dialogue['frame']) if dialogue else dict(lastQuery=history)
    if changed_metric and not reference:
        context=dict(currentMetric=changed_metric['requested'],currentMetricFamilies=changed_metric['families'],regions=declared_regions,
            instruction='本轮明确更换统计指标。仅按本轮问题生成计划，不继承旧指标、旧排名或旧输出列。')
    from query_semantics import risk_level
    if (not dialogue and not reference and not population and unbounded_group_summary(q) and risk_level(q)
        and not re.search(r'刚才|上一|上述|这些|这批|原来|沿用|不变|相同|同一口径',q)):
        # A complete new report supplies its own measure and scope. Old row-list
        # plans are not requirements for this report; persisted references stay.
        context=dict(newCompleteGroupReport=True)
    if population:
        context['boundPopulation']=population
        context['populationInstruction']=('本轮需要独立声明对象和比较基准；previous沿用ID，reselect沿用选样方法。' if multi_scope else '本轮samples被程序限定为该集合；只改变本轮统计方式，不能要求用户重填筛选条件。')
    SCHEMA = service.schema
    plan_prompt=('You interpret a Chinese database question into a compact query contract. Do NOT write SQL or answers yet. '
      'Database rows already exist. Values such as averages, totals and ratios must be computed from columns, NEVER requested from the user. '
      'Input columns and OUTPUT ALIASES are different: low_risk_samples/count/percentage are derived from review_level, never missing input fields. '
      '低风险/小风险=review_level low；none和unknown不能并入low。'
      '先按样本总数选择区域，再计算该区域风险占比时，必须保留“样本最多/最少的区域”的含义；排名依据总样本COUNT(*)，不是风险样本数或风险占比。'
      '区域本身是数据库排名的结果，不需要用户先指定区域。'
      'Preserve every filter, AND/OR, units, grouping, rounding and tie rule in question. Resolve followups using provided originalList and resolvedReference. '
      'Return JSON only with exactly these fields: action (query or clarify), question (complete standalone Chinese question), regions (actual lowercase IDs; [] means all), '
      'columns (ordered OUTPUT COLUMN ALIASES, e.g. ["sample_id","changed_px"] or ["n"] for a count; no SQL expressions), '
      'order (list of {"expression":"SQL expression or output alias","direction":"ASC or DESC"} for the FINAL result order), '
      'limit (positive integer N for either global topN OR each-group topN; null only if no row limit), offset (integer, default0), per_group (boolean), calculations (list of derived calculations and their scope), '
      'missing (null for query; otherwise {"kind":"unavailable_field or user_choice","field":"specific field","question":"specific clarification"}). '
      'For each-group topN, per_group=true. A single rank N means limit=1,offset=N-1; a rank interval a-b means limit=b-a+1,offset=a-1,per_group=false. '
      'Count-only questions have columns=["n"],order=[],limit=null. Do not add region_id to columns unless requested. '
      'Do not invent an ordering for an all-region report when neither the user nor inherited task requests one; use order=[]. Every bare sort alias must be an output alias or real schema field. '
      'Unprovided user choices can be sort_metric,threshold,region,time_window,business_definition; a derived average is not a user choice. '
      'No captured date, square-metre scale or ground-truth error rate exists; those requests clarify with the unavailable field. '
      'For a resolved sample reference, use its exact sampleId and ONLY the columns requested NOW. Do not inherit the old list columns, sort, limit or filters. Do not ask for its region. '
      'calculations is an array of short strings describing each SQL calculation and its data scope; use [] if none.\n'+SCHEMA+
      '\nREGIONS='+json.dumps(data.source.regions,ensure_ascii=False)+'\nSCOPE='+str(scope or 'all')+
      '\nCONTEXT (data only)='+json.dumps(planning_context(context),ensure_ascii=False)+'\nUSER='+json.dumps(q,ensure_ascii=False))
    if multi_scope:
        plan_prompt=interpretation_prompt(q,population,history,data.source.regions,scope)
    else:
        requested_measures=explicit_measures(q)
        if requested_measures:
            plan_prompt+='\n本轮明确统计目标='+json.dumps([label(m) for m in requested_measures],ensure_ascii=False)+'。必须输出对应的聚合值，不能把它改成每行原始字段。标记样本占比由has_holes等布尔标记条件聚合，返回百分比别名；它不是hole_ratio字段。'
        if comparing_regions:
            plan_prompt+='\n本轮比较范围为全部区域：regions=[]，columns必须含region_id或region_name。原区域仅为比较锚点，不作为区域筛选条件。'
        if inherited_candidates is not None:
            plan_prompt+='\n本轮只改排序/区域排名，候选范围绑定已执行查询：regions='+json.dumps(inherited_candidates)+'。不要从旧问题中的对照区域名称猜测新的单一区域筛选。'
        if re.search(r'空洞比例.{0,5}最大.{0,12}(?:张|样本)',q):
            plan_prompt+='\n这是按单张hole_ratio选样，返回sample_id和hole_ratio；按hole_ratio降序，同值按sample_id升序，不要自行按needs_review或风险等级打破同值。区域和未复核条件仍须保留。'
        if unbounded_group_summary(q):
            plan_prompt+='\n本轮是按区域汇总、每个区域一行：per_group=false,limit=null,offset=0。GROUP BY分组与“每组各取前N条”不是同一任务，不要继承旧排名任务的标记或数量。'
        from query_semantics import regional_total_rank, risk_level
        if regional_total_rank(q) and (risk_level(q) or dialogue):
            plan_prompt+='\n本轮必须按全部样本COUNT(*)选择一个区域，再计算问题指定的统计目标（数量保持数量，均值保持均值，占比保持占比）。order首项直接使用COUNT(*)，方向'+regional_total_rank(q)+'；limit必须1，offset为0，per_group=false。不要使用未定义的sample_count。保留原问题中的“样本最多/最少的区域”，返回区域标识与所问指标。calculations只用自然语言说明范围，不要写未经验证的SQL公式。'
    if service.task_type == 'building_segmentation' and not multi_scope:
        from task_profiles import contract_prompt
        plan_prompt = contract_prompt(SCHEMA,q,data.source.regions,scope,planning_context(context))
    c=None;prepared=None;error='';previous_contract=''
    for attempt in range(2):
        try:
            repair_instruction=('只输出selection、reference、comparison、clarification，修正错误并保留本轮其他条件；无需追问时clarification:null，不要增加action或missing字段。' if multi_scope else '仅修正上述问题并输出完整JSON，保持全部原始条件。没有缺失条件就action=query、missing=null。')
            output=invoke(plan_prompt+ ('\n上轮查询要求JSON='+previous_contract+'\n需修正的校验错误='+error+'\n'+repair_instruction if error else ''),'contract',1024)
            previous_contract=str(output.get('text','') if isinstance(output,dict) else output)[:3000]
            known_regions={r['id'] for r in data.source.regions}
            c=parse_contract(compile_contract(output,q,population,known_regions) if multi_scope else output,known_regions,source_question=None if multi_scope else q,schema=SCHEMA)
            if (c['action']=='query' and 'sample_id' in c['columns']
                and re.search(r'(?:多少|几(?:个|张|组))\s*样本|样本(?:组)?(?:的)?(?:数量|总数)',q)
                and not re.search(r'样本\s*ID|sample_id|列出|找出|明细|每张|每条|逐条',q,re.I)):
                raise ContractError('本轮要求计数，不是逐行列出sample_id。请返回COUNT(*)的计数别名；按区域统计时保留区域列。')
            if dialogue and c['action']=='query':
                # Context has already been resolved and measure slots bound.
                # Do not let the output-formatting pass rewrite that task again.
                c['question']=q
                c.setdefault('normalizations',[]).append('question_from_bound_task')
                from query_semantics import regional_total_rank
                direction=regional_total_rank(q)
                if direction:
                    c.update(order=[dict(expression='COUNT(*)',direction=direction)],limit=1,offset=0,per_group=False)
                    c['normalizations'].append('selection_from_bound_task')
            from query_semantics import validate_rank_interpretation
            validate_rank_interpretation(q,c)
            if dialogue and c['action']=='query':validate_question(c['question'],dialogue['required'])
            if not multi_scope:normalize_group_summary(c,q)
            if not multi_scope:
                from query_updates import validate_rank_contract
                validate_rank_contract(q,c)
            if not multi_scope and c['action']=='query':
                if len(explicit)==1 and c['limit'] is not None and not re.search(r'每(?:个)?区域|各(?:个)?区域|每组|分组|分别',q):
                    if c['per_group']:c.setdefault('normalizations',[]).append('single_region_global_topn')
                    c['per_group']=False
                sort_requested=r'排序|升序|降序|从大到小|从小到大|前[0-9一二三四五六七八九十]|第[0-9一二三四五六七八九十]|最[多少大小高低]|排名|按'
                sort_question=dialogue['original'] if dialogue else q
                c['orderRequired']=bool(re.search(sort_requested,sort_question) or regional_total_rank(q) or (dialogue and re.search(r'其余|不变|仍然',dialogue['original']) and re.search(sort_requested,dialogue['frame']['question'])))
                if c['per_group']:
                    c['rankOrder']=[o for o in c['order'] if sqlglot.parse_one(o['expression'],read='sqlite').name not in {'region_id','region_name'}]
                    c['orderRequired']=bool(re.search(r'最终|结果.{0,8}按|输出.{0,8}按|跨区域|全局排序',q))
                if comparing_regions:
                    c['regions']=[]
                    c.setdefault('normalizations',[]).append('all_regions_from_comparison')
                    if not any(x in {'region_id','region_name'} for x in c['columns']):raise ContractError('跨区域比较须返回region_id或region_name，不能只返回原区域的一个统计值。')
                if inherited_candidates is not None:
                    c['regions']=inherited_candidates
                    c.setdefault('normalizations',[]).append('candidate_regions_from_executed_query')
                if declared_regions and dialogue and not comparing_regions:
                    c['regions']=declared_regions
                    c.setdefault('normalizations',[]).append('regions_from_current_message')
                if dialogue and dialogue.get('extreme_direction'):
                    c.update(limit=1,offset=0,per_group=False)
                    c.setdefault('normalizations',[]).append('one_region_from_metric_extremum')
                edit_direction=(dialogue or {}).get('sort_direction') or (dialogue or {}).get('extreme_direction')
                if edit_direction and (not c['order'] or c['order'][0]['direction']!=edit_direction):
                    raise ContractError('本轮必须按保留的统计指标'+edit_direction+'排序，请在order中写明指标输出别名与方向。')
                if any(m['operation'].startswith('flag_') for m in explicit_measures(q)) and any(re.search(r'(?:low|medium|high)_risk',x) for x in c['columns']):
                    raise ContractError('有空洞标记的样本不是低/中/高风险样本，请用has_holes_count、sample_count、has_holes_percentage等输出别名，不要将标记计数命名为low_risk_samples。')
            if explicit and not set(explicit)<=set(c['regions']):raise ContractError('regions遗漏用户明确指定的区域：'+','.join(explicit))
            if scope and any(r!=scope for r in c['regions']):raise ContractError('查询要求超出当前选定区域。')
            if c['action']=='query' and multi_scope:
                if not c.get('scopes'):raise ContractError('缺少scopes。它是你须输出的内部计划，请按原提示填写selection、reference、comparison，不能向用户索要scopes或均值。')
                prepared=prepare_scopes(c['scopes'],history,scope,q)
                if c['scopes']['comparison']['output']=='rows':c['order']=result_order(prepared)
            elif c.get('scopes'):
                raise ContractError('本轮未请求独立范围比较，不应增加scopes。')
            if c['action']=='clarify' and c['missing']['kind']=='user_choice':
                if c['missing']['field']=='region' and (explicit or scope or reference):raise ContractError('区域已明确或样本ID已绑定，不能重复追问区域。')
                if c['missing']['field']=='sort_metric' and c['order']:raise ContractError('排序指标已经给出，不能重复追问。')
            result['attempts'].append(dict(stage='contract',status='planned' if c['action']=='query' else 'clarification',sql=None,queryContract=c,modelGeneratedContract=previous_contract))
            break
        except Exception as exc:
            error=str(exc)[:400];result['attempts'].append(dict(stage='contract',status='rejected',sql=None,error=error,contractPreview=previous_contract));c=None
            if isinstance(exc,StaleData):break
    if c is None:
        save_scope_attempt()
        result.update(status='failed',answer='未能形成有效的查询要求，未执行SQL。',error=error);return finish()
    result['queryContract']=c
    if service.task_type=='building_segmentation':
        from task_profiles import building_count_requested
        if building_count_requested(q):c['requiresBuildingEstimate']=True
        c['buildingCountRegions']=[scope] if scope else [r['id'] for r in data.source.regions]
        if building_total:
            c['buildingTotal']=True;c['regions']=[]
        if re.fullmatch(r'(?:请)?(?:列出|查询|查|找出)?(?:本批|这批|全批)(?:全部|所有)?(?:图片的)?建筑量(?:最多|最高|最少|最低)的?[0-9一二两三四五六七八九十]+张(?:图片|影像|图像)(?:是哪些|有哪些)?[。？?！!]*',input_question.strip()):c['unfilteredBuildingRank']=True
    if (service.task_type=='building_segmentation' and 'sample_id' in c['columns'] and c.get('limit')==1
        and c.get('offset',0)>0 and not re.search(r'均值|平均|高于|低于|超过|大于|小于',q)):
        c['simpleSampleRank']=True
    bare_mean_question=re.split(r'[？?。；;]',q,1)[0].strip()
    if (service.task_type=='building_segmentation' and not scope and re.fullmatch(
        r'(?:请)?(?:统计|计算)?(?:本批全部|全批|所有|全部)(?:的)?(?:图片|影像|样本)?(?:的)?(?:平均)?建筑(?:像素)?(?:占比|比例)(?:的)?(?:平均值|均值|平均)?(?:是多少)?',bare_mean_question)):
        c['includeZeroForeground']=True
    if c['action']=='clarify':
        try:save_pending(c['missing']['question'],missing=c['missing'])
        except Exception as exc:
            result.update(status='failed',answer='追问上下文保存失败。',error=str(exc)[:400]);return finish()
        return clarify(c['missing']['question'],'model_contract')
    if prepared:
        result['referenceStats']=reference_stats(prepared,data)
        if result['referenceStats']['value'] is None and c['scopes']['comparison']['operator']!='share':
            save_scope_attempt()
            return clarify('比较基准没有该指标的有效数值，无法判断高于或低于均值；未把未知基准当作零。请选择有有效指标的比较范围。','reference_data_guard')
    # Reuse the tested domain/region/NULL semantics instead of a second divergent
    # schema-only prompt. The contract adds checks; it does not replace SQL rules.
    sql_prompt=(service._prompt(c['question'],data,scope,None)+
      '\n按下列CONTRACT精确生成SQL。已知统计条件不必追问；若实际缺失必要字段，应指出缺失字段，不能删除该条件改算其他数值。只返回sql/clarification两个字段的JSON。'
      '输出严格限于指定列及其顺序，计算表达式请AS指定别名，不要增加区域ID、样本ID或排名列。'
      '保持所有筛选、分组筛选、舍入、分母范围和外层ORDER BY。全局分页只使用最外层LIMIT/OFFSET一次。'
      'CONTRACT的order为空时不增加ORDER BY，limit为null时不增加LIMIT；严格遵守输出别名，不使用未声明的排序别名。'
      '每组前N条使用单表samples的直接ROW_NUMBER分区排名子查询或CTE，再按rn<=N取值，不能全局截断。'
      '若有绑定样本引用，主查询必须FROM samples并用AND条件绑定该sample_id，仅返回本轮所问字段。'
      '\nCONTRACT='+json.dumps(c,ensure_ascii=False)+'\nRESOLVED_REFERENCE='+json.dumps(reference,ensure_ascii=False)+
      '\nORIGINAL_USER='+json.dumps(q,ensure_ascii=False))
    if population:
        sql_prompt+='\nBOUND_POPULATION='+json.dumps(population,ensure_ascii=False)+'\n本轮samples输入由程序冻结为上述集合，直接对samples执行本轮要求的统计。不要添加或猜测新的筛选条件；不得改读regions。'
    if prepared:
        field_schema=SCHEMA.split('CREATE TABLE samples(',1)[1].split(');',1)[0]
        logical_schema='CREATE TABLE selected_samples('+field_schema+');\nCREATE TABLE reference_samples('+field_schema+');'
        precision=re.search(r'保留\s*([0-9零一二两三四五六])\s*位(?:小数)?',q)
        decimals=_ordinal(precision[1]) if precision else None
        calculation_contract=dict(columns=c['columns'],order=c['order'],decimal_places=decimals,
            reference={k:c['scopes']['reference'][k] for k in ('metric','aggregate')},comparison=c['scopes']['comparison'])
        sql_prompt=('你是SQLite分析查询生成器。只返回JSON {"sql":"完整查询","clarification":null}。'
            '程序提供两个互相独立的只读逻辑表：selected_samples为本轮统计对象，reference_samples为比较基准。'
            '两表字段完全相同；只能引用这两个逻辑表，禁止引用samples或regions。每条查询必须有FROM selected_samples。'
            '选样、区域、筛选和前N张已经由程序完成，不需要你重做。主查询必须直接FROM selected_samples，禁止包一层选样子查询或重复LIMIT。'
            '所有统计和比较SQL由你编写，程序只绑定数据范围。'
            '基准直接通过括号内标量子查询读取reference_samples，不需要CTE或JOIN。'
            '不要对两个逻辑表再添加区域、排名、阈值筛选；程序已按scopes提供完整集合。'
            'count输出COUNT(*)并在WHERE比较对象指标与基准；rows输出对象字段并在WHERE比较；'
            'percentage比较占比用100.0*SUM(CASE WHEN 对象指标与基准比较 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0)，'
            'share集合占比用100.0*COUNT(*)/NULLIF((SELECT COUNT(*) FROM reference_samples),0)。'
            'NULL未知值不能当零；decimal_places非null时对百分比ROUND到该位数；输出列与排序必须遵循计算契约。\n'
            +logical_schema+'\nreference_samples与selected_samples字段完全相同。\n本轮只需生成统计计算SQL：'+json.dumps(calculation_contract,ensure_ascii=False))
    if c['limit'] is not None and not c['per_group']:
        sql_prompt+='\n本轮是全局或单区域分页，请直接SELECT所需列 FROM samples WHERE原始筛选 ORDER BY指定字段 LIMIT '+str(c['limit'])+' OFFSET '+str(c['offset'])+'。无需CTE或ROW_NUMBER，不要将排序字段误作分区字段。'
    if c['per_group']:
        sql_prompt+='\n每组窗口内部的ORDER BY必须遵循rankOrder；允许在最后加sample_id升序处理同值。外层只是结果展示，可按区域、窗口排名使用的指标稳定排列，不要用外层排序替代窗口内的选样排序。rankOrder='+json.dumps(c.get('rankOrder'),ensure_ascii=False)
    if c.get('simpleSampleRank'):
        sql_prompt+='\n本轮是直接全局分页。例如建筑占比第二高：SELECT sample_id,sample_name,foreground_ratio FROM samples ORDER BY foreground_ratio DESC LIMIT 1 OFFSET 1。排名序号仅用于OFFSET，绝不添加MAX/MIN子查询或排除第一名的WHERE条件。依CONTRACT保留所需字段和实际筛选；ORDER BY字段数量、顺序和方向严格按CONTRACT，不额外增加sample_id排序。'
    if c.get('includeZeroForeground'):
        sql_prompt+='\n本轮明确全批均值，直接SELECT AVG(foreground_ratio) AS 所需别名 FROM samples。包括0值，不加WHERE，不做NULLIF，不排除foreground_pixels=0。'
    computable_mean=bool(re.search(r'平均|均值',q) and re.search(r'变化像素|changed_px|对象数|对象数量|object_count',q) and re.search(r'本区域|该区域|全库|全部样本|所有样本',q))
    if computable_mean:
        sql_prompt+='\n问题中的均值可以从现有数据计算，不缺数据字段或计算条件。按明确的区域/全库口径，在独立标量子查询中计算AVG(所问指标)，再与主查询样本比较；不要向用户索要均值。'
    required_measures=explicit_measures(q)
    percentage_precision=('\nSQLite的整数除法会截断。浮点转换必须发生在除法之前：先将分子乘100.0再除以分母，'
        '或先CAST分子AS REAL再除。不能先算整数分子/整数分母，再在外层乘100.0；那时精度已经丢失。'
        '标记样本百分比可写100.0*SUM(CASE WHEN 标记字段=1 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0)。'
        '保留原始分子条件、全部样本分母及舍入要求。')
    for marker in [m for m in required_measures if m['operation']=='flag_percentage']:
        sql_prompt+='\n本轮统计标记样本百分比，标记字段为'+marker['field']+'，1代表有标记。分子是标记为1的样本数，分母是当前统计范围全部COUNT(*)；用浮点条件聚合输出百分比。单张面积比例不能代替标记样本占比，不能逐行输出面积字段；不要先WHERE该标记=1再除以筛选后的COUNT(*)。'
        sql_prompt+=percentage_precision
    if comparing_regions:
        sql_prompt+='\n最终执行范围：全部区域，按region_id分组并返回区域标识。原问题开头的区域是上一轮对照点，不是本轮WHERE筛选；禁止再用WHERE region_id=旧区域。保留上一轮非区域条件，不要将均值换成标记样本占比。上一轮WHERE（仅非区域部分应延续）='+json.dumps(dialogue['frame'].get('filters'),ensure_ascii=False)+'。严格遵循CONTRACT的最终ORDER BY与输出别名。'
    from query_semantics import regional_total_rank, risk_level, derived_risk_field
    regional_risk=bool(regional_total_rank(q) and (risk_level(q) or dialogue) and not prepared and not population and not reference)
    aggregate_risk=bool(risk_level(q) and any(x in {'region_id','region_name'} for x in c['columns']) and not prepared and not population and not reference)
    if regional_risk or aggregate_risk:
        sql_prompt=('Generate one read-only SQLite SELECT from the real schema below. '
          'Return JSON with exactly sql and clarification; executable SQL means clarification=null. '
          'Output EXACTLY the CONTRACT column aliases in order, no extra region_name or sample_id. '
          +('First rank regions by total sample COUNT(*), then report the requested measures inside that region. ' if regional_risk else 'Report the requested measure for each requested region; do not select a top region unless requested. ')+
          'Preserve quantity versus percentage: when a count is asked, return the conditional count without percentage conversion. '
          'Use direct GROUP BY region_id; no CTE/subquery is needed. '
          +('The regional selection order is COUNT(*) (or its total-count alias). ' if regional_risk else 'ORDER BY exactly the OUTPUT_REQUIREMENTS order, including percentage ordering when requested; do not replace it with count ordering. ')+
          'Risk appears only inside conditional aggregation, never as a sample WHERE filter. '
          'low/小风险/低风险=low, medium/中风险=medium, high/高风险=high; none and unknown are not low. '
          'The percentage is 100.0 times conditional risk count DIVIDED BY COUNT(*); multiply BEFORE division to avoid integer division. '
          'Keep all samples including none and unknown in that region denominator. '
          'Output aliases such as low_risk_samples are calculated, not missing columns. '
          +('Do not ask which region: find it by the requested ranking. ' if regional_risk else 'Include each requested region; a zero risk count must not remove a region. ')+
          'Only JOIN regions on region_id when region_name is requested. '
          +SCHEMA+'\nREGIONS='+json.dumps(data.source.regions,ensure_ascii=False)
          +'\nOUTPUT_REQUIREMENTS='+json.dumps({k:([] if k=='order' and not c.get('orderRequired',True) else c[k]) for k in ('columns','order','limit','offset')},ensure_ascii=False)
          +'\nORIGINAL_USER='+json.dumps(q,ensure_ascii=False))
        if ('未复核' in q)!=('已复核' in q):
            reviewed_value=0 if '未复核' in q else 1
            sql_prompt+='\nREQUIRED_SAMPLE_FILTERS='+json.dumps(dict(reviewed=reviewed_value))+'. Apply these in WHERE before GROUP BY. needs_review is an independent original quality flag; it must not replace reviewed or be added when not requested. Only the risk condition belongs inside CASE; the reviewed filter belongs in WHERE.'
        if dialogue:
            task=dict(measures=dialogue['required'],candidate_regions=c['regions'] or 'all_visible_regions',
                region_selection=dict(metric='total_sample_count',direction=regional_total_rank(q)) if regional_total_rank(q) else None)
            sql_prompt+='\nAUTHORITATIVE_STATISTICAL_TASK='+json.dumps(task,ensure_ascii=False)+\
                '\nThe statistical task is already resolved. Generate exactly these measure operations and risk predicates. The region selection count and the reported risk count are separate aggregates. Do not reinterpret a negated ranking criterion as cancelling the reported risk measure. Candidate regions constrain the samples source before regional aggregation; use its region_id in WHERE.'
    error='';sql=''
    if c.get('unfilteredBuildingRank'):
        sql_prompt+='\n本轮明确为本批全部图片排名：不要WHERE，不要HAVING，不要复核筛选，不要estimated_building_count>0。0也是真实建筑量，不能丢弃。仅按每行estimated_building_count排序并分页，不需要SUM或GROUP BY。'
    if c.get('buildingTotal'):sql_prompt+='\n本轮为当前批次建筑量总量，只输出SUM(estimated_building_count)这一列，不要列表、GROUP BY、ORDER BY、WHERE或LIMIT。'
    for attempt in range(2):
        generated_sql='';generated_output=''
        try:
            if error:
                repair=(sql_prompt+'\n修复下面查询，保留用户原始条件；模型额外添加的旧区域限制须按本轮范围移除。\n上次SQL='+json.dumps(sql,ensure_ascii=False)+'\n校验错误='+error) if prepared or regional_risk or aggregate_risk or comparing_regions else ('你是SQLite查询修复器。修正下面查询的校验错误，只返回JSON {"sql":"修复后的完整SELECT","clarification":null}。'
                    '真实数据已经存在。只能使用表结构里的字段名。若必要字段确实不存在，sql=null并在clarification中指出该字段；不能丢条件、不能替换成其他指标。'
                    '保持全部筛选、统计口径和返回列；返回列中的计算别名须由真实字段聚合产生，不是缺失的原始字段。排序必须在最终SELECT的ORDER BY中体现。'
                    '全局分页只用最外层LIMIT/OFFSET一次，不要用ROW_NUMBER；每组排名才使用分区ROW_NUMBER。\n'+SCHEMA+
                    '\n实际区域ID与名称='+json.dumps(data.source.regions,ensure_ascii=False)+
                    '\n查询要求='+json.dumps(c,ensure_ascii=False)+'\n绑定引用='+json.dumps(reference,ensure_ascii=False)+
                    '\n原用户问题='+json.dumps(q,ensure_ascii=False)+'\n待修复SQL='+json.dumps(sql,ensure_ascii=False)+
                    '\n校验错误='+error+'\n返回修复后的SQL JSON；仅在必要字段确实缺失时澄清。')
                if any(m['operation']=='flag_percentage' for m in required_measures):repair+=percentage_precision
            else:repair=sql_prompt
            if c.get('unfilteredBuildingRank'):
                repair+='\n最终要求：每张图片直接取estimated_building_count，保留0值；不使用WHERE、GROUP BY或HAVING。上一条SQL中的任何筛选都不是本轮用户要求。'
            if c.get('buildingTotal'):repair+='\n最终要求：当前批次建筑量总量，只返回SUM(estimated_building_count)，不能沿用上一轮的图片列表、排序或筛选。'
            output=invoke(repair,'sql',1536)
            generated_output=str(output.get('text','') if isinstance(output,dict) else output)[:16000]
            sql,clarification=parse_output(output)
            generated_sql=sql or ''
            if clarification:
                asks=re.search(r'(?:请|需要|能否).*(?:提供|告知|输入|明确).*(?:平均|均值|总数|总量|占比)',clarification,re.S)
                repeats=bool(reference and re.search(r'提供.*(?:区域|样本.?ID)',clarification,re.I))
                derived=any(derived_risk_field(v) for v in re.findall(r'\b(?:low|medium|high)_risk(?:_[a-z]+)*\b',clarification,re.I))
                rank_region=bool(regional_total_rank(q) and re.search(r'明确|提供|指定',clarification) and re.search(r'区域|地区',clarification))
                missing_mean=bool(computable_mean and re.search(r'缺少|缺失|未提供|没有',clarification) and re.search(r'平均|均值',clarification) and not re.search(r'日期|拍摄时间|采集时间|平方米|地面尺度|真值|阈值',clarification))
                if derived:raise ContractError('low/medium/high风险数量和占比由review_level条件聚合计算，不需要额外风险字段；输出别名不代表原始字段缺失。')
                if asks or repeats or rank_region or missing_mean:raise ContractError('均值、计数、占比及按排名选出的区域应由SQL查询，不能要求用户提供这些可计算的值。')
                physical_columns=set(re.findall(r'\b([a-z_]+)\s+(?:TEXT|INTEGER|REAL)\b',SCHEMA))
                missing_slot=decisions.sql_missing_slot(clarification,set(c['columns'])|physical_columns)
                if missing_slot is None:
                    mentioned=set(re.findall(r'[a-z_][a-z0-9_]*',clarification.lower())) & physical_columns
                    if mentioned:
                        existing=[table+'.'+field for table,body in re.findall(r'CREATE TABLE\s+(\w+)\s*\((.*?)\);',SCHEMA,re.S)
                                  for field in re.findall(r'\b([a-z_]+)\s+(?:TEXT|INTEGER|REAL)\b',body) if field in mentioned]
                        raise ContractError('字段已存在于真实表结构：'+', '.join(existing)+'。NULL表示未知值，不表示字段不存在；保留SQL聚合的NULL，不得替换为0、改指标或要求用户补充该字段。按已确认条件生成SQL。')
                    raise ContractError('这是SQL生成或格式错误，应由模型修复，不能要求用户补充业务条件：'+clarification)
                save_pending(clarification,missing=missing_slot)
                result['attempts'].append(dict(stage='sql',status='clarification',sql=None,modelGeneratedOutput=generated_output))
                return clarify(clarification,'sql_model')
            from query_semantics import validate_semantics
            if prepared:
                validate_sql(sql,c,scope,None)
                result['semanticChecks']=validate_semantics(sql,q,c,[],scope,bound_population=True)+['independent_population_scopes','comparison_expression']
                sql=read_sql(validate_and_expand(sql,prepared))
            else:
                sql=read_sql(sql);validate_sql(sql,c,scope,reference,total_sample_rank=regional_risk)
                if dialogue:validate_measures(sql,dialogue['required'])
                if dialogue and (dialogue.get('sort_direction') or dialogue.get('extreme_direction')):
                    validate_measure_order(sql,[dialogue['sort_measure']] if dialogue.get('sort_measure') else dialogue['required'],dialogue.get('sort_direction') or dialogue.get('extreme_direction'))
                if required_measures:validate_measures(sql,required_measures)
                if comparing_regions or inherited_candidates==[]:
                    comparison_tree=sqlglot.parse_one(sql,read='sqlite')
                    where=comparison_tree.args.get('where')
                    region_where=where.copy() if where else None
                    if region_where:
                        # Listing every visible ID is redundant but semantically
                        # valid. Reject a restricted subset, not this tautology.
                        all_regions={r['id'] for r in data.source.regions}
                        for term in list(region_where.find_all(exp.In)):
                            if isinstance(term.this,exp.Column) and term.this.name=='region_id' and not term.args.get('query') and all(isinstance(v,exp.Literal) and v.is_string for v in term.expressions) and {v.this for v in term.expressions}==all_regions:
                                term.replace(exp.true())
                    if region_where and any(col.name in {'region_id','region_name'} for col in region_where.find_all(exp.Column)):
                        raise ContractError('本轮对比各区域，上一轮区域只作对照；不能继续用WHERE筛选原区域或部分区域。请按各区域分组返回同一指标。')
                result['semanticChecks']=validate_semantics(sql,q,c,explicit,scope,
                    bound_population=bool(population),reference=bool(reference))
            if population and not prepared:
                validate_analysis(sql,q)
                sql=read_sql(bind_population(sql,population))
            executed=data.execute(sql,region_id=scope)
            if reference and 'sample_id' in executed['columns'] and any(r['sample_id']!=reference['sampleId'] for r in executed['rows']):raise ContractError('返回结果不符合已绑定样本ID。')
            if review_version(service.reviewed_provider())!=data.review_version or Snapshot(service.source_path).source_version!=data.version:raise StaleData('执行期间数据或复核记录更新，请重新提问。')
            next_population=saved_population(prepared,data) if prepared else (history or {}).get('population') if reference else derive_population(sql,executed,q,population)
            selected_population=next_population
            if prepared and c['scopes']['comparison']['output']=='rows':next_population=returned_population(prepared,executed)
            if not c.get('orderRequired',True):
                actual_tree=sqlglot.parse_one(executed['sql'],read='sqlite')
                actual_order=[dict(expression=o.this.sql(dialect='sqlite'),direction='DESC' if o.args.get('desc') else 'ASC') for o in (actual_tree.args.get('order') or exp.Order()).expressions]
                if c['order']!=actual_order:result['inferredOrderAlignment']=dict(modelOrder=c['order'],executedOrder=actual_order)
                c['order']=actual_order
            state={**remember(history,q,c,executed,next_population),'analysis':analysis_state(sql),
                  'review_population':reference_population(reference) if reference else next_population}
            if prepared:
                state['scope_frame']=c['scopes']
                result['scopePlan']=c['scopes']
                if c['scopes']['selection']['mode']=='reselect' and c['scopes']['comparison']['output']!='rows':
                    original=_list(history,True)
                    state['last_list']=dict(list_id=uuid.uuid4().hex,question=q,contract=c,ordered_ids=next_population['ordered_ids'])
                    if original:state['root_list']=original
                result['selectionSampleIds']=(selected_population.get('ordered_ids') or selected_population.get('filters',{}).get('values',[]))
                result['statisticalSelectionCount']=selected_population.get('sample_count')
            save(state)
            if next_population and next_population.get('supported'):
                result['population']={**next_population,'inherited':prepared['scopes']['selection']['mode']=='previous' if prepared else bool(population)}
            if reference:result['population']={**reference_population(reference),'inherited':True}
            result['attempts'].append(dict(stage='sql',status='executed',sql=executed['sql'],modelGeneratedSQL=generated_sql))
            result.update(status='completed',sqlQuery=executed['sql'],sqlRows=executed['rows'],sqlColumns=executed['columns'],
                modelGeneratedSQL=generated_sql,programChanges=['scoped_populations'] if prepared else ['bound_population'] if population else [],
                sqlTruncated=executed['truncated'],grounded=True,answer=f'已校验并执行模型生成的只读SQL，返回{len(executed["rows"])}行'+('（已截断）' if executed['truncated'] else '')+'。')
            if c['limit'] is not None and not c['per_group'] and not (state.get('taskFrame') or {}).get('regionSelection'):
                from query_updates import report_task,result_summary
                ranked=report_task(state)
                if ranked:
                    result['answer']=result_summary(executed['rows'],ranked,dict(action='update_query',changes=dict(rank=ranked['rank'])),data.source.regions)[0]
            result.pop('error',None);return finish()
        except Exception as exc:
            error=str(exc)[:400];result['attempts'].append(dict(stage='sql',status='rejected',sql=sql,error=error,modelGeneratedSQL=generated_sql[:16000],modelGeneratedOutput=generated_output))
            if isinstance(exc,StaleData):break
    result.update(status='failed',answer='SQL未满足查询要求或执行失败，未返回结果。',error=error)
    save_scope_attempt()
    return finish()
