"""Small explicit task constraints; no SQL templates or inferred answer values."""
import re
import copy

import sqlglot
from sqlglot import exp

from model_sql import QueryError
from query_updates import NUMBER,number


def segmentation_clarification(goal):
    temporal=re.search(r'变化(?:像素|面积)?(?:比例|占比|率)|changed_ratio|changed_px',goal,re.I)
    if temporal:
        revised=re.sub(r'变化(?:像素|面积)?(?:比例|占比|率)|changed_ratio|changed_px','建筑覆盖比例',goal,flags=re.I)
        return dict(status='needs_clarification',originalGoal=goal,
            answer='当前批次是单图建筑分割，没有两期影像的变化比例。你是想按建筑覆盖比例给图片排名，并查看该图片的建筑量吗？确认后才替换指标；如果要分析真实变化比例，请先切换到变化检测批次。',
            clarificationKind='segmentation_metric',options=[dict(id='foreground_ratio',label='按建筑覆盖比例继续',goal=revised)])
    if re.search(r'新增|新建|拆除|拆了|消失|变了|变化数量|前后时相',goal):
        return dict(status='needs_clarification',originalGoal=goal,
            answer='当前是单图建筑分割，无法判断新增、拆除或前后变化。请切换到变化检测批次，或明确改查当前建筑量及覆盖比例。',clarificationKind='unavailable_temporal_data',options=[])
    return None


def sample_rank_request(goal,task_type):
    """A single ordinal image plus its attributes, not a regional comparison."""
    if not re.search(r'图片|影像|图像|样本',goal) or re.search(r'每.{0,3}区域|各区域|区域.{0,6}(?:排名|第)|第'+NUMBER+r'(?:高|低)?(?:的)?区域',goal):return None
    ordinal=re.search(r'(?:排名?\s*第?|排在第?|第)\s*('+NUMBER+r')',goal)
    if not ordinal:return None
    start=number(ordinal[1])
    if not 1<=start<=10000:raise QueryError('图片排名必须为1至10000的有效名次。')
    # Keep the metric before the image noun separate from attributes requested
    # after it, e.g. rank by coverage and return the same image's building count.
    head=re.split(r'图片|影像|图像|样本',goal,maxsplit=1)[0]
    if not re.search(r'比例|占比|率|像素|建筑量|建筑数量|_ratio|_count|_pixels|changed_px',head):
        head=re.split(r'以及|并返回|并给出|并统计|同时返回|及其|及它|及他',goal,maxsplit=1)[0]
    seg=task_type=='building_segmentation'
    if re.search(r'建筑(?:像素)?(?:覆盖)?(?:占比|比例)|建筑覆盖率|foreground_ratio',head):field='foreground_ratio'
    elif re.search(r'变化(?:占比|比例|率)|changed_ratio',head):field='changed_ratio'
    elif re.search(r'建筑像素|foreground_pixels',head):field='foreground_pixels'
    elif re.search(r'变化像素|changed_px',head):field='changed_px'
    elif re.search(r'建筑量|建筑数量|estimated_building_count',head):field='estimated_building_count'
    else:return None
    if seg and field.startswith('changed_'):return None
    return dict(field=field,start=start,end=start,direction='ASC' if re.search(r'第'+NUMBER+r'低|最低|升序|从低到高',head) else 'DESC',
                buildingCount=bool(seg and re.search(r'建筑量|建筑(?:物)?数量|(?:多少|几)栋',goal)),
                noExtraFilters=not bool(re.search(r'复核|大于|小于|超过|低于|高于|至少|至多|包含|不含|排除|仅|只看',goal)))


def validate_sample_rank(sql,spec,scope_regions=()):
    tree=sqlglot.parse_one(sql,read='sqlite')
    source=(tree.args.get('from_') or exp.From()).this
    if (not isinstance(tree,exp.Select) or not isinstance(source,exp.Table) or source.name!='samples'
        or tree.ctes or tree.find(exp.Subquery) or tree.find(exp.Window) or tree.find(exp.AggFunc)
        or any(tree.args.get(k) for k in ('joins','group','having','distinct'))):
        raise QueryError('图片排名必须逐张读取samples，不能改为区域分组、建筑量汇总或跨区域比较；请用一个直接SELECT返回所选图片及其属性。')
    required={'sample_id','sample_name',spec['field']}|({'estimated_building_count'} if spec.get('buildingCount') else set())
    projected=set();aliases={}
    for p in tree.expressions:
        core=p.this if isinstance(p,exp.Alias) else p
        if isinstance(core,exp.Column):
            projected.add(core.name)
            if isinstance(p,exp.Alias):aliases[p.alias]=core.name
    if not required<=projected:raise QueryError('图片排名结果必须返回同一张图片的'+','.join(sorted(required))+'，不能只返回区域或汇总值。')
    order=tree.args.get('order');first=order.expressions[0] if order and order.expressions else None
    if (first is None or not isinstance(first.this,exp.Column) or aliases.get(first.this.name,first.this.name)!=spec['field']
        or ('DESC' if first.args.get('desc') else 'ASC')!=spec['direction']):
        raise QueryError('图片排名的第一排序指标必须是未四舍五入的'+spec['field']+' '+spec['direction']+'，不能换成建筑数量或区域指标。')
    if not any(isinstance(o.this,exp.Column) and aliases.get(o.this.name,o.this.name)=='sample_id' and not o.args.get('desc') for o in order.expressions[1:]):
        raise QueryError('图片排名同值时须按sample_id ASC稳定排序，保证第三张可以复核。')
    limit=tree.args.get('limit');offset=tree.args.get('offset')
    if not limit or limit.expression.sql()!='1' or (offset.expression.sql() if offset else '0')!=str(spec['start']-1):
        raise QueryError(f'图片排名第{spec["start"]}只能返回1张：LIMIT 1 OFFSET {spec["start"]-1}，不是前N张或区域排名。')
    if spec.get('noExtraFilters',True):
        def allowed(n):
            if isinstance(n,exp.Paren):return allowed(n.this)
            if isinstance(n,exp.And):return allowed(n.this) and allowed(n.expression)
            if isinstance(n,exp.Not) and isinstance(n.this,exp.Is):
                # PublicQueryData inserts reviewed as int(bool), never NULL.
                return isinstance(n.this.this,exp.Column) and n.this.this.name in {spec['field'],'reviewed'} and isinstance(n.this.expression,exp.Null)
            if isinstance(n,exp.EQ):
                a,b=n.this,n.expression
                if isinstance(a,exp.Literal):a,b=b,a
                return isinstance(a,exp.Column) and a.name=='region_id' and isinstance(b,exp.Literal) and b.is_string and bool(scope_regions) and set(scope_regions)<={b.this}
            if isinstance(n,exp.In) and not n.args.get('query'):
                return isinstance(n.this,exp.Column) and n.this.name=='region_id' and all(isinstance(v,exp.Literal) and v.is_string for v in n.expressions) and bool(scope_regions) and set(scope_regions)<={v.this for v in n.expressions}
            return False
        if tree.args.get('where') and not allowed(tree.args['where'].this):raise QueryError('图片排名没有请求复核筛选或非零筛选，不能额外添加；0也是有效比例。')
    return ['sample_ranking_grain','same_sample_attributes','requested_rank_metric','literal_rank_pagination']


def prepare_sample_plan(value,spec):
    """Append the declared stable tie key, preserving the model's original SQL."""
    if not spec or value.get('clarification'):return value
    value=copy.deepcopy(value)
    for step in value.get('steps') or []:
        if not isinstance(step,dict):continue
        sql=step.get('sql') or ''
        try:tree=sqlglot.parse_one(sql,read='sqlite')
        except sqlglot.errors.ParseError:continue
        source=(tree.args.get('from_') or exp.From()).this
        if (not isinstance(tree,exp.Select) or not isinstance(source,exp.Table) or source.name!='samples' or tree.ctes
            or tree.find(exp.Subquery) or tree.find(exp.AggFunc) or tree.find(exp.Window)
            or any(tree.args.get(k) for k in ('joins','group','having','distinct'))):continue
        order=tree.args.get('order')
        if not order or not order.expressions:continue
        aliases={p.alias:p.this.name for p in tree.expressions if isinstance(p,exp.Alias) and isinstance(p.this,exp.Column)}
        first=order.expressions[0]
        if not isinstance(first.this,exp.Column) or aliases.get(first.this.name,first.this.name)!=spec['field']:continue
        if any(isinstance(o.this,exp.Column) and aliases.get(o.this.name,o.this.name)=='sample_id' for o in order.expressions[1:]):continue
        order.append('expressions',exp.Ordered(this=exp.column('sample_id',table=source.alias_or_name),desc=False))
        step['modelGeneratedSQL']=sql;step['sql']=tree.sql(dialect='sqlite')
        step['programChanges']=['stable_sample_id_tie_break']
    return value


def sample_plan_problem(value,spec,scope_regions):
    if not spec:return None
    steps=value.get('steps')
    if not isinstance(steps,list) or len(steps)!=1:return '本次只需一个图片排名查询，同时返回该图片的排序指标及建筑量；不要拆成区域排名或比较步骤。'
    try:validate_sample_rank(steps[0].get('sql',''),spec,scope_regions)
    except (ValueError,sqlglot.errors.ParseError) as exc:return str(exc)
    return None
