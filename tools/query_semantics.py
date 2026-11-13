"""Bounded semantic checks independent of the model's interpretation.

These prove a few explicit constraints on supported SQL shapes, not arbitrary
natural-language equivalence. No reference answers or replacement SQL live here.
"""
import re
import sqlglot
from sqlglot import exp


class SemanticError(ValueError):
    pass


def _possible(predicate, field, aliases, universe):
    """Conservative value set: unknown expressions may allow the whole universe."""
    if predicate is None:return set(universe)
    if isinstance(predicate,exp.Paren):return _possible(predicate.this,field,aliases,universe)
    if isinstance(predicate,exp.And):
        return _possible(predicate.this,field,aliases,universe)&_possible(predicate.expression,field,aliases,universe)
    if isinstance(predicate,exp.Or):
        return _possible(predicate.this,field,aliases,universe)|_possible(predicate.expression,field,aliases,universe)
    def column(node):
        return isinstance(node,exp.Column) and node.name.lower()==field and (not node.table or node.table.lower() in aliases)
    def literal(node):
        if not isinstance(node,exp.Literal):return None
        return str(node.this) if node.is_string else int(node.this) if str(node.this).isdigit() else None
    if isinstance(predicate,exp.EQ):
        a,b=predicate.this,predicate.expression
        if column(b):a,b=b,a
        value=literal(b)
        if column(a) and value is not None:return {value}&set(universe)
    if isinstance(predicate,exp.In) and column(predicate.this) and not predicate.args.get('query'):
        values=[literal(v) for v in predicate.expressions]
        if all(v is not None for v in values):return set(values)&set(universe)
    # Do not negate an unknown upper bound: that would manufacture a proof.
    return set(universe)


def _samples_sources(query):
    source=query.args.get('from_')
    relations=([source.this] if source else [])+[j.this for j in query.args.get('joins') or []]
    return [t for t in relations if isinstance(t,exp.Table) and t.name.lower()=='samples']


def _region_aliases(query,source_alias):
    """Propagate region keys only through mandatory INNER equijoin terms."""
    def terms(node):
        if isinstance(node,exp.Paren):return terms(node.this)
        if isinstance(node,exp.And):return terms(node.this)+terms(node.expression)
        return [node] if node is not None else []
    edges=[]
    for join in query.args.get('joins') or []:
        if join.args.get('side') or str(join.args.get('kind') or '').upper() not in {'','INNER'}:continue
        for term in terms(join.args.get('on')):
            if isinstance(term,exp.EQ) and all(isinstance(n,exp.Column) and n.name.lower()=='region_id' and n.table for n in (term.this,term.expression)):
                edges.append({term.this.table.lower(),term.expression.table.lower()})
    aliases={source_alias.lower()}
    for _ in range(len(edges)):
        for edge in edges:
            if edge&aliases:aliases|=edge
    return aliases


def regional_total_rank(question):
    """Recognize an explicit unqualified sample-total extremum, not risk ranking."""
    if not re.search(r'区域|地区',question):return None
    for match in re.finditer(r'(?:样本|影像|记录)(?:的)?(?:总数量|总数|数量|总量|数|量)?(?:为)?(?:选|选出|选取|挑|取)?(最多|最大|最少|最小)',question):
        prefix=question[:match.start()].rstrip()
        if re.search(r'(?:风险|复核|空洞|碎片)(?:的)?$',prefix):continue
        return 'DESC' if match[1] in {'最多','最大'} else 'ASC'
    return None


def risk_level(question):
    levels={level for pattern,level in [(r'高风险','high'),(r'中风险','medium'),(r'[低小]风险','low')] if re.search(pattern,question)}
    return next(iter(levels)) if len(levels)==1 else None


def derived_risk_field(field):
    return bool(re.fullmatch(r'(?:low|medium|high)[_ -]*risk(?:[_ -]*(?:samples?|count|number|num|pct|percentage|percent|ratio|rate))*',field.strip('`" '),re.I))


def validate_rank_interpretation(question,contract):
    """Reject a rewritten question that moves risk from the result to ranking."""
    direction=regional_total_rank(question)
    if not direction:return
    if contract['action']=='clarify':
        missing=contract['missing']
        if missing['field']=='region':
            raise SemanticError('区域是按样本总数排名查询得到的结果，不是用户缺失的选择；请生成查询。')
        return
    if regional_total_rank(contract['question'])!=direction:
        raise SemanticError('原问题按区域全部样本总数排名，必须在独立问题中保留“样本最多/最少的区域”；不能改成风险数量或风险占比最多。')
    if risk_level(question):
        if risk_level(contract['question'])!=risk_level(question) or bool(re.search(r'占比|比例|百分',question))!=bool(re.search(r'占比|比例|百分',contract['question'])):
            raise SemanticError('选出区域后仍须计算原问题指定的风险统计；风险等级、数量与占比不能改变。')
    if risk_level(question) or re.search(r'平均|均值|总和|合计',question):
        order=contract['order']
        if not order or order[0]['direction']!=direction:
            raise SemanticError('区域排名方向必须对应原问题的总样本最多/最少。')
        if contract['limit']!=1 or contract['offset'] or contract['per_group']:
            raise SemanticError('本轮选样本总数最多/最少的一个区域，须limit=1、offset=0、per_group=false，不能返回全部区域。')
        first=sqlglot.parse_one(order[0]['expression'],read='sqlite')
        if not (isinstance(first,exp.Count) and isinstance(first.this,exp.Star)) and order[0]['expression'] not in contract['columns']:
            raise SemanticError('总样本排名直接用COUNT(*)，或者声明已返回的总样本数别名；不要使用未定义的sample_count排序别名。')
        if not re.search(r'只(?:要|返回|给出).{0,5}(?:占比|比例|百分)',question) and not any(c in {'region_id','region_name'} for c in contract['columns']):
            raise SemanticError('选区域再报风险占比时，应返回区域标识region_id或region_name，不能丢掉区域分组。')


def _validate_total_rank_risk(tree,question):
    """Bounded proof for direct grouped conditional-risk aggregation.

    Complex equivalent CTE shapes are deliberately sent back for SQL repair.
    No query or answer is generated by this check.
    """
    direction=regional_total_rank(question);level=risk_level(question)
    if not direction or (not level and not re.search(r'平均|均值|合计|总和',question)):return False
    sources=_samples_sources(tree)
    if len(sources)!=1 or tree.ctes or tree.find(exp.Subquery):
        raise SemanticError('当前总样本排名与风险占比校验支持直接按区域聚合：FROM samples，按region_id分组，用条件聚合计算风险占比；不要使用嵌套排名来源。')
    aliases={sources[0].alias_or_name.lower()}
    def unalias(node):return node.this if isinstance(node,exp.Alias) else node
    def own_column(node,name):return isinstance(node,exp.Column) and node.name.lower()==name and (not node.table or node.table.lower() in aliases)
    # Every inserted sample_id is a non-null normalized string. With the proven
    # one-to-one INNER region join below, COUNT(sample_id) equals COUNT(*).
    def count_all(node):return isinstance(node,exp.Count) and (isinstance(node.this,exp.Star) or own_column(node.this,'sample_id')) and not node.args.get('expressions')
    outputs={e.alias.lower():e.this for e in tree.expressions if isinstance(e,exp.Alias)}
    order=(tree.args.get('order') or exp.Order()).expressions
    first=order[0].this if order else None
    if isinstance(first,exp.Column) and not first.table:first=outputs.get(first.name.lower(),first)
    if not order or not count_all(first) or bool(order[0].args.get('desc'))!=(direction=='DESC'):
        raise SemanticError('区域必须按全部样本COUNT(*)排名，不能按风险条件计数或风险占比排名。')
    group=(tree.args.get('group') or exp.Group()).expressions
    region_keys={'region_id','region_name'}
    if not any(isinstance(g,exp.Column) and g.name=='region_id' for g in group) or any(not isinstance(g,exp.Column) or g.name not in region_keys for g in group):
        raise SemanticError('必须按region_id聚合每个区域全部样本，不能按风险等级拆组。')
    where=tree.args.get('where')
    if tree.args.get('having') or (where and (not list(where.find_all(exp.Column)) or any(c.name!='region_id' for c in where.find_all(exp.Column)))):
        raise SemanticError('区域总样本排名不能先筛风险或其他样本条件；风险只进入条件聚合分子。')
    relations=([tree.args['from_'].this] if tree.args.get('from_') else [])+[j.this for j in tree.args.get('joins') or []]
    regions=[r for r in relations if isinstance(r,exp.Table) and r.name.lower()=='regions']
    if len(relations)>2 or len(regions)>1 or any(not isinstance(r,exp.Table) or r.name.lower() not in {'samples','regions'} for r in relations):
        raise SemanticError('区域汇总只支持单一samples来源，及可选的单一regions名称来源。')
    for join in tree.args.get('joins') or []:
        on=join.args.get('on');pairs=(on.this,on.expression) if isinstance(on,exp.EQ) else (None,None)
        if (not regions or join.args.get('side') or
            str(join.args.get('kind') or '').upper() not in {'','INNER'} or
            not all(isinstance(p,exp.Column) and p.name=='region_id' for p in pairs) or
            {p.table.lower() for p in pairs}!=aliases|{regions[0].alias_or_name.lower()}):
            raise SemanticError('区域名称只可通过samples.region_id=regions.region_id连接，不能改变样本计数。')
    if not level:return True
    def numeric(node,value):
        return isinstance(node,exp.Literal) and not node.is_string and str(node.this) in {str(value),str(float(value))}
    def indicator(node,count=False):
        if not isinstance(node,exp.Case) or node.this is not None or len(node.args.get('ifs') or [])!=1:return False
        branch=node.args['ifs'][0];condition=branch.this
        if not isinstance(condition,exp.EQ):return False
        a,b=condition.this,condition.expression
        if isinstance(a,exp.Literal):a,b=b,a
        correct=own_column(a,'review_level') and isinstance(b,exp.Literal) and b.is_string and b.this==level
        fallback=node.args.get('default')
        return correct and numeric(branch.args.get('true'),1) and (fallback is None or isinstance(fallback,exp.Null) if count else numeric(fallback,0))
    def conditional_count(node):
        return isinstance(node,exp.Sum) and indicator(node.this) or isinstance(node,exp.Count) and indicator(node.this,True)
    def numerator(node):
        while isinstance(node,exp.Paren):node=node.this
        if not isinstance(node,exp.Mul):return False
        def scaled(a,b):
            if not numeric(a,100):return False
            floating=isinstance(a,exp.Literal) and '.' in str(a.this)
            if isinstance(b,exp.Cast) and b.args['to'].sql().upper() in {'REAL','FLOAT','DOUBLE'}:floating=True;b=b.this
            return floating and conditional_count(b)
        return scaled(node.this,node.expression) or scaled(node.expression,node.this)
    if not re.search(r'占比|比例|百分',question):
        if not any(conditional_count(unalias(p)) for p in tree.expressions):
            raise SemanticError('选出区域后仍须返回指定review_level的条件计数，不能只返回区域总样本数，也不能将数量改成占比。')
        return True
    valid=False
    for projection in tree.expressions:
        value=unalias(projection)
        if isinstance(value,exp.Round):value=value.this
        while isinstance(value,exp.Paren):value=value.this
        if not isinstance(value,exp.Div):continue
        denominator=value.expression
        while isinstance(denominator,exp.Paren):denominator=denominator.this
        if isinstance(denominator,exp.Nullif) and numeric(denominator.expression,0):denominator=denominator.this
        # Require percentage scaling and conditional count, with full group count
        # in the denominator; none/unknown rows remain in that denominator.
        if count_all(denominator) and numerator(value.this):valid=True
    if not valid:raise SemanticError('风险占比须用100.0乘指定review_level的条件计数，再除以该区域COUNT(*)；none/unknown保留在分母且不能并入低风险。')
    return True


def validate_semantics(sql,question,contract,explicit_regions,physical_scope=None,*,bound_population=False,reference=False):
    tree=sqlglot.parse_one(sql,read='sqlite')
    checks=[]
    if not isinstance(tree,exp.Select):return checks
    sources=_samples_sources(tree)
    aliases={t.alias_or_name.lower() for t in sources}
    predicate=tree.args.get('where')
    predicate=predicate.this if predicate else None
    # Only the main relation is checked. A scalar aggregate subquery can have a
    # deliberately different scope (e.g. the global denominator).
    if explicit_regions and sources and not bound_population and not reference:
        allowed=set(explicit_regions)
        if physical_scope:allowed &= {physical_scope}
        # Keep every actual literal as well as OTHER. Intersecting an excluded
        # literal with an incomplete universe would turn a real escape into an
        # impossible branch and manufacture a proof.
        universe=allowed|{str(v.this) for v in tree.find_all(exp.Literal) if v.is_string}|{'__OTHER_REGION__'}
        for source in sources:
            possible=_possible(predicate,'region_id',_region_aliases(tree,source.alias_or_name),universe)
            if not possible<=allowed:
                raise SemanticError('主查询区域约束可能被OR分支绕过或遗漏。请让region_id约束每个samples来源及所有主查询分支，标量子查询保持其原统计范围。')
        checks.append('explicit_main_region')
    if re.search(r'尚未复核|未复核|没有复核|还没复核|未完成复核',question) and sources and not bound_population:
        for source in sources:
            if _possible(predicate,'reviewed',{source.alias_or_name.lower()},{0,1})!={0}:
                raise SemanticError('未复核必须约束reviewed=0，needs_review是原始质量提示，不能代替人工复核状态；条件需覆盖每个samples来源及所有OR分支。')
        checks.append('unreviewed_status')
    match=re.search(r'保留\s*([0-9零一二两三四五六])\s*位(?:小数)?',question)
    if match:
        digits=int(match[1]) if match[1].isdigit() else {'零':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6}[match[1]]
        for projection in tree.expressions:
            for node in projection.walk():
                if not isinstance(node,(exp.Avg,exp.Div)) or node.find_ancestor(exp.Select) is not tree:continue
                parent=node.parent;rounded=False
                while parent is not None and parent is not tree:
                    if isinstance(parent,exp.Round):
                        decimals=parent.args.get('decimals')
                        rounded=decimals is not None and decimals.sql()==str(digits)
                        break
                    parent=parent.parent
                if not rounded:raise SemanticError(f'明确要求保留{digits}位小数，输出的均值或比例必须使用ROUND(...,{digits})。')
        checks.append('explicit_decimal_precision')
    global_count=bool(re.search(r'(?:分母.{0,12}(?:全库|所有区域|全部区域)|占.{0,16}(?:所有区域|全部区域|全库).{0,10}(?:样本|记录).{0,5}(?:总数|数量))',question))
    if global_count:
        found_division=False
        for projection in tree.expressions:
            divisions=[n for n in projection.find_all(exp.Div) if n.find_ancestor(exp.Select) is tree]
            for division in divisions:
                found_division=True
                valid=False
                denominator=division.expression
                while isinstance(denominator,exp.Paren):denominator=denominator.this
                if isinstance(denominator,exp.Nullif) and isinstance(denominator.expression,exp.Literal) and denominator.expression.sql()=='0':
                    denominator=denominator.this
                    while isinstance(denominator,exp.Paren):denominator=denominator.this
                if isinstance(denominator,exp.Subquery):
                    q=denominator.this
                    if isinstance(q,exp.Select) and len(_samples_sources(q))==1 and not any(q.args.get(k) for k in ('where','group','having','joins','limit','offset','with_')):
                        core=q.expressions[0] if len(q.expressions)==1 else None
                        if isinstance(core,exp.Alias):core=core.this
                        valid=isinstance(core,exp.Count) and isinstance(core.this,exp.Star)
                if not valid:raise SemanticError('全库样本总数分母必须独立读取未筛选的samples并COUNT(*)，不能使用已受区域或风险条件限制的COUNT(*)。')
        if not found_division:raise SemanticError('本轮要求全库占比，输出必须实际包含除以独立全库样本总数的计算。')
        checks.append('global_sample_denominator')
    if risk_level(question) and re.search(r'占比|比例|百分',question):
        def floating(expression):
            for n in expression.walk():
                if n.find_ancestor(exp.Select) is not tree:continue
                if isinstance(n,exp.Literal) and not n.is_string and '.' in str(n.this):return True
                if isinstance(n,exp.Cast) and n.args['to'].sql().upper() in {'REAL','FLOAT','DOUBLE'}:return True
                if isinstance(n,exp.Avg):return True
            return False
        for projection in tree.expressions:
            for division in projection.find_all(exp.Div):
                if division.find_ancestor(exp.Select) is tree and not (floating(division.this) or floating(division.expression)):
                    raise SemanticError('风险占比在除法发生前必须转为浮点，例如先乘100.0或CAST AS REAL；整数相除后再乘100会截断结果。')
        checks.append('floating_risk_division')
    if not bound_population and not reference and _validate_total_rank_risk(tree,question):
        checks.append('regional_total_rank_and_risk_share')
    return checks
