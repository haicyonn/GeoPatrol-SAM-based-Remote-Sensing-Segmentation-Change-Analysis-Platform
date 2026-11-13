"""Version-bound sample populations, independent of a model's next calculation.

SQL is still model-authored. For a resolved population, AST composition confines
every samples input to the saved selector before execution. This is a supported
query-shape mechanism, not a proof of arbitrary SQL semantic equivalence.
"""
from __future__ import annotations

import copy
import re

import sqlglot
from sqlglot import exp


class PopulationError(ValueError):
    pass


DEICTIC = re.compile(r'这些(?:样本|图|切片|记录)?|这批|这[两二三四五六七八九十\d]+(?:张|个|幅)|上述(?:样本|结果)|该批|它们|刚才(?:筛出|选出|查出|那批|那几)|上一轮(?:筛出|选出|查出)')
EDIT = re.compile(r'放宽|放松|再加|加上|并入|合并|扩大|扩展|改看|换成|取消.{0,8}(?:条件|限制)|去掉.{0,8}(?:条件|限制)|把.{0,20}(?:改成|改为)|(?:区域|下限|上限|阈值).{0,8}(?:改成|改为)')


def resolve_population(question, history):
    if not DEICTIC.search(question):
        return None
    population = (history or {}).get('population')
    if not population or not population.get('supported'):
        reason = (population or {}).get('reason') or '当前会话还没有可绑定的样本集合'
        raise PopulationError(reason + '；请先明确区域和筛选条件查询，再用“这些样本”追问。')
    if EDIT.search(question):
        raise PopulationError('这次要求改变原筛选范围，请明确写出修改后的完整条件；不会把旧集合误当成放宽后的集合。')
    if re.search(r'全库|全部区域|所有区域|本区域(?:的)?(?:全部|总|平均|均值)|区域总数|(?:占|与|和|比).{0,10}(?:本区域|原区域|该区域|(?:LEVIR|WHU|CLCD|GZ.?CD|UAV).{0,6}(?:总|全部|平均|均值))', question,re.I):
        raise PopulationError('本轮同时涉及原样本集合和外部统计范围，请分别查询两者或明确写出完整比较条件。')
    return copy.deepcopy(population)


def describe_filter(node):
    if node is None:
        return {'kind': 'all'}
    if isinstance(node, exp.Paren):
        return describe_filter(node.this)
    if isinstance(node, (exp.And, exp.Or)):
        return {'kind': node.key, 'arguments': [describe_filter(node.this), describe_filter(node.expression)]}
    if isinstance(node, exp.Not):
        return {'kind': 'not', 'argument': describe_filter(node.this)}
    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Is)) and isinstance(node.this, exp.Column):
        value = node.expression
        if isinstance(value, (exp.Literal, exp.Null, exp.Boolean)):
            return {'field': node.this.name, 'operator': node.key, 'value': value.sql(dialect='sqlite')}
    return {'kind': 'expression', 'sql': node.sql(dialect='sqlite')}


def analysis_state(sql):
    tree = sqlglot.parse_one(sql, read='sqlite')
    output = []
    for expression in tree.expressions:
        entry = {'output': expression.alias_or_name}
        aggregate = next(expression.find_all(exp.AggFunc), None)
        if aggregate:
            entry.update(operation=aggregate.key, expression=aggregate.this.sql(dialect='sqlite'))
        rounded = expression.find(exp.Round)
        if rounded:
            entry['decimal_places'] = rounded.args.get('decimals').sql() if rounded.args.get('decimals') else '0'
        output.append(entry)
    return {'outputs': output}


def reference_population(reference):
    sid=reference['sampleId']
    selector=exp.select('sample_id').from_('samples').where(exp.column('sample_id').eq(exp.Literal.string(sid)))
    return {'supported':True,'kind':'displayed_samples','selector_sql':selector.sql(dialect='sqlite'),
            'filters':{'field':'sample_id','operator':'in','values':[sid]},'region_ids':[],
            'origin_question':f'已核验样本ID：{sid}' if reference.get('source')=='verified_sample_id' else f'上一轮列表第{reference["position"]}张：{sid}','sample_count':1}


def derive_population(sql, executed, question, inherited=None):
    tree = sqlglot.parse_one(sql, read='sqlite')
    unsupported = lambda reason: {'supported': False, 'reason': reason}
    if not isinstance(tree, exp.Select) or tree.args.get('group') or tree.args.get('having') or tree.args.get('joins') or tree.args.get('with_'):
        return unsupported('上一轮是分组、连接或复杂查询，不能自动确定“这些样本”对应哪批记录')
    source = tree.args.get('from_')
    source = source.this if source else None
    if not ((isinstance(source, exp.Table) and source.name.lower() == 'samples') or
            (inherited and isinstance(source, exp.Subquery))):
        return unsupported('上一轮没有可复用的单一样本来源')
    regions = sorted({str(v.this) for v in tree.find_all(exp.Literal) if v.is_string and str(v.this) in {'levir', 'whu', 'clcd', 'gzcd', 'uav'}})
    if inherited:
        regions = sorted(set(regions) | set(inherited.get('region_ids', [])))
    expressions = [e.this if isinstance(e, exp.Alias) else e for e in tree.expressions]
    if inherited and tree.args.get('where') is None and any(e.find(exp.AggFunc) for e in expressions):
        return copy.deepcopy(inherited)
    if any(e.find(exp.Case) or e.find(exp.Filter) or e.find(exp.Distinct) or e.find(exp.Subquery)
           for e in expressions):
        return unsupported('上一轮包含条件聚合、去重或投影子查询，无法唯一确定参与统计的样本；请用WHERE明确筛选条件')
    sample_projection = next((e for e in expressions if isinstance(e, exp.Column) and e.name == 'sample_id'), None)
    if 'sample_id' in executed['columns'] and sample_projection is not None:
        if executed.get('truncated'):
            return unsupported('上一轮样本列表被截断，无法把展示部分默认为完整样本集合')
        ids = [r['sample_id'] for r in executed['rows']]
        selector = exp.select('sample_id').from_('samples').where(
            exp.column('sample_id').isin(*[exp.Literal.string(str(s)) for s in ids]) if ids else exp.false())
        from query_scopes import selection_recipe
        recipe=selection_recipe(sql) if not inherited else inherited.get('selection_recipe')
        return {'supported': True, 'kind': 'displayed_samples', 'selector_sql': selector.sql(dialect='sqlite'),
                'selection_recipe':recipe,
                'filters': {'field': 'sample_id', 'operator': 'in', 'values': ids},
                'region_ids': regions, 'origin_question': question[:1000], 'sample_count': len(ids)}
    if not any(e.find(exp.AggFunc) for e in expressions):
        return unsupported('上一轮未返回可绑定的样本ID或样本集合统计')
    if tree.args.get('limit') or tree.args.get('offset') or tree.args.get('distinct'):
        return unsupported('上一轮聚合包含截断或去重，无法无歧义恢复样本集合')
    where = tree.args.get('where')
    if inherited and where is None:
        return copy.deepcopy(inherited)
    alias = source.alias_or_name or 'samples'
    selector = tree.copy()
    selector.set('expressions', [exp.column('sample_id', table=alias)])
    selector.set('order', None)
    text = selector.sql(dialect='sqlite')
    if len(text) > 3500:
        return unsupported('样本集合定义超过长度预算，请重新明确完整筛选条件')
    filters = describe_filter(where.this if where else None)
    if inherited:
        filters = {'kind': 'and', 'arguments': [inherited['filters'], filters]}
    return {'supported': True, 'kind': 'filtered_population', 'selector_sql': text, 'filters': filters,
            'region_ids': regions, 'origin_question': inherited['origin_question'] if inherited else question[:1000]}


def bind_population(sql, population):
    """Constrain actual data inputs even if generated SQL omits every old filter."""
    tree = sqlglot.parse_one(sql, read='sqlite')
    if any(c.alias_or_name.lower() in {'samples', 'regions'} for c in tree.find_all(exp.CTE)):
        raise PopulationError('继承集合时不能用CTE覆盖samples或regions。')
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = list(tree.find_all(exp.Table))
    if any(t.db or t.catalog or t.name.lower() not in {'samples'} | ctes for t in tables):
        raise PopulationError('本轮只查询已绑定样本集合，不能改读regions或其他数据来源。')
    inputs = [t for t in tables if t.name.lower() == 'samples']
    if not inputs:
        raise PopulationError('本轮必须实际读取已绑定的样本集合。')
    selector = sqlglot.parse_one(population['selector_sql'], read='sqlite')
    # Work only on original table nodes; do not recursively rewrite our own scope.
    for table in inputs:
        alias = table.args.get('alias') or exp.TableAlias(this=exp.to_identifier(table.name))
        relation = exp.select('*').from_('samples').where(exp.column('sample_id').isin(query=selector.copy()))
        table.replace(exp.Subquery(this=relation, alias=alias.copy()))
    return tree.sql(dialect='sqlite', pretty=True)


def validate_analysis(sql, question):
    """Check explicit simple aggregate/rounding requests outside model interpretation."""
    metric = 'changed_px' if re.search(r'变化像素|变化面积', question) else 'object_count' if re.search(r'对象数|对象数量|目标数', question) else None
    operation = exp.Avg if '平均' in question or '均值' in question else exp.Sum if re.search(r'总变化像素|合计|总和|求和', question) else exp.Max if '最大值' in question else exp.Min if '最小值' in question else None
    if operation is None or metric is None:
        return
    tree = sqlglot.parse_one(sql, read='sqlite')
    candidates=[]
    for expression in tree.expressions:
        core=expression.this if isinstance(expression,exp.Alias) else expression
        if isinstance(core,exp.Round):core=core.this
        if isinstance(core,operation) and isinstance(core.this,exp.Column) and core.this.name==metric:
            candidates.append(core)
    if not candidates:
        raise PopulationError('本轮必须计算' + metric + '的' + ('AVG' if operation is exp.Avg else 'SUM') + '，不能替换统计对象或计算方式。')
    match = re.search(r'保留\s*([0-9零一二两三四五六])\s*位(?:小数)?', question)
    if not match:
        return
    digits = int(match[1]) if match[1].isdigit() else {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6}[match[1]]
    for aggregate in candidates:
        parent = aggregate.parent
        while parent is not None and not isinstance(parent, exp.Select):
            decimals = parent.args.get('decimals') if isinstance(parent, exp.Round) else None
            if decimals is not None and decimals.sql() == str(digits):
                return
            parent = parent.parent
    raise PopulationError(f'本轮明确要求保留{digits}位小数，必须使用ROUND(...,{digits})。')
