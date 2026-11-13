"""Conservative checks of structured numeric claims, not semantic entailment.

Supported assertions are equalities over known sample/report fields, with their
original units. Ratios may be rendered as percentages and report shares may be
computed from named counts. Other numeric prose needs a literal excerpt in the
locally cited source. Passing these checks does not establish that narrative
claims follow from the sources, or that the underlying facts are correct.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


_ARABIC = r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?'
_CHINESE = r'负?[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二两三四五六七八九]+)?'
_NAMED_CHINESE = (r'(?:对象数量|对象数|变化像素|变化比例|置信度|总切片|总样本|优先复核|待复核)'
                  r'\s*[为是：:]?\s*' + _CHINESE + r'(?=\s*(?:[，,。；;！？!?\n\[\]]|$))')
NUMBER = re.compile(r'(?<![A-Za-z0-9_])(?:百分之(?:' + _ARABIC + '|' + _CHINESE + ')|'
                    + _NAMED_CHINESE + '|' + _ARABIC + '|' + _CHINESE
                    + r'(?=\s*(?:像素|px|平方米|平方公里|公顷|亩|个(?:对象|目标|连通域|样本)|张|幅|条|对象)))')
CITATION = re.compile(r'\[(\d+)\]')
CLAUSE_BREAK = re.compile(r'[，,；;。！？!?\n]')
SENTENCE_BREAK = re.compile(r'[。！？!?\n]')
FIELD_METRICS = {
    'changed_px': ('changed_px', 'pixels'), 'changedPx': ('changed_px', 'pixels'),
    'object_count': ('object_count', 'objects'), 'objectCount': ('object_count', 'objects'),
    'changed_ratio': ('changed_ratio', 'ratio'), 'changedRatio': ('changed_ratio', 'ratio'),
    'confidence': ('confidence', 'ratio'),
    'totalCount': ('total_count', 'samples'), 'total_rows': ('total_count', 'samples'),
    'changedCount': ('changed_count', 'samples'), 'changed_rows': ('changed_count', 'samples'),
    'reviewCount': ('review_count', 'samples'), 'review_rows': ('review_count', 'samples'),
    'correctionCount': ('correction_count', 'records'),
}
SAMPLE_METRICS = {'changed_px', 'object_count', 'changed_ratio', 'confidence'}
METRIC_TERMS = {
    'changed_px': ('变化像素', '变化面积', '变化', 'changed_px', 'changedPx'),
    'object_count': ('对象数量', '对象数', '对象', '目标数', '连通域数量', 'object_count', 'objectCount'),
    'changed_ratio': ('变化比例', '变化占比', '变化率', '变化百分比', 'changed_ratio', 'changedRatio'),
    'confidence': ('置信度', 'confidence'),
    'total_count': ('总切片', '总样本', '总计', '合计', '共有', '共', 'totalCount', 'total_rows'),
    'changed_count': ('变化切片', '变化样本', 'changedCount', 'changed_rows'),
    'review_count': ('优先复核', '待复核', '需复核', '复核样本', '复核切片', 'reviewCount', 'review_rows'),
    'correction_count': ('人工修订', '人工修正', '人工更正', '修订记录', 'correctionCount'),
    'risk:high': ('高风险',), 'risk:medium': ('中风险',),
    'risk:low': ('低风险', '常规样本'), 'risk:none': ('无风险',),
}


def validation_scope() -> dict:
    """Describe the checks without implying a full correctness judgment."""
    return {'checks': ['citation_range', 'sample_metric_numeric_bindings', 'units',
                       'literal_source_numeric_excerpts', 'basic_forbidden_absolutes'],
            'semanticEntailment': False, 'fullSemanticCorrectness': False,
            'numericNotation': 'Arabic numbers and common Chinese numeric measure phrases',
            'supportedConversions': ['ratio_to_percent', 'snapshot_count_share']}


@dataclass(frozen=True)
class _Fact:
    sample: str | None
    metric: str
    value: Decimal
    unit: str


def _decimal(value) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).replace(',', ''))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _number_value(value) -> Decimal | None:
    """Parse detected Arabic numbers and common Chinese integer/decimal forms."""
    value = str(value).removeprefix('百分之')
    number = _decimal(value)
    if number is not None:
        return number
    match = re.search(_CHINESE + '$', value)
    if not match:
        return None
    token = match[0]
    negative = token.startswith('负')
    integer, _, fraction = token.removeprefix('负').partition('点')
    digits = dict(zip('零〇一二两三四五六七八九', (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    units = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
    if all(char in digits for char in integer):
        total = int(''.join(str(digits[char]) for char in integer))
    else:
        total = section = current = 0
        for char in integer:
            if char in digits:
                current = digits[char]
            elif units[char] < 10000:
                section += (current or 1) * units[char]
                current = 0
            elif char == '万':
                total += (section + current) * 10000
                section = current = 0
            else:
                total = (total + section + current) * units[char]
                section = current = 0
        total += section + current
    decimal_text = str(total) + ('.' + ''.join(str(digits[char]) for char in fraction) if fraction else '')
    return Decimal(('-' if negative else '') + decimal_text)


def _structured_facts(facts):
    records, aliases = [], {}
    terms = {key: list(value) for key, value in METRIC_TERMS.items()}

    def add(sample, metric, value, unit):
        number = _decimal(value)
        if number is not None:
            records.append(_Fact(sample, metric, number, unit))

    def visit(node, inherited=None):
        if isinstance(node, list):
            for item in node:
                visit(item, inherited)
            return
        if not isinstance(node, dict):
            return
        identifiers = [str(node[key]) for key in ('sample_id', 'sampleId', 'sample_stem', 'stem')
                       if node.get(key) is not None and str(node[key])]
        if any(key in node for key in ('changed_px', 'changedPx', 'object_count', 'objectCount',
                                      'changed_ratio', 'changedRatio', 'confidence')):
            identifiers += [str(node[key]) for key in ('id', 'sample_name', 'name') if node.get(key)]
        sample = identifiers[0] if identifiers else inherited
        for identifier in identifiers:
            aliases[identifier] = sample
        for key, (metric, unit) in FIELD_METRICS.items():
            if key in node:
                add(sample if metric in SAMPLE_METRICS else None, metric, node[key], unit)
        # Only a report's explicitly named counts form its share denominator.
        total = _decimal(node.get('totalCount', node.get('total_rows')))
        if total is not None and total > 0:
            for name, metric in (('changedCount', 'changed_share'), ('reviewCount', 'review_share')):
                numerator = _decimal(node.get(name))
                if numerator is not None:
                    add(None, metric, numerator / total, 'ratio')
        if 'correctionCount' not in node and isinstance(node.get('correctionEntries'), list):
            add(None, 'correction_count', len(node['correctionEntries']), 'records')
        for key, value in node.items():
            if key in {'riskBuckets', 'reasonBuckets', 'eventBuckets', 'scaleBuckets'} and isinstance(value, list):
                category = {'riskBuckets': 'risk', 'reasonBuckets': 'reason',
                            'eventBuckets': 'event', 'scaleBuckets': 'scale'}[key]
                for bucket in value:
                    if not isinstance(bucket, dict):
                        continue
                    label = str(bucket.get('key') or bucket.get('label') or '')
                    if not label:
                        continue
                    metric = category + ':' + label
                    add(None, metric, bucket.get('count'), 'samples')
                    if bucket.get('ratio') is not None:
                        add(None, metric + '_share', bucket['ratio'], 'ratio')
                    for alias in (bucket.get('label'), bucket.get('key')):
                        if alias and len(str(alias)) > 1:
                            terms.setdefault(metric, []).append(str(alias))
            else:
                visit(value, sample)

    visit(facts)
    return records, aliases, terms


def _mask(text, pattern):
    return re.sub(pattern, lambda match: ' ' * len(match[0]), text)


def _alias_spans(text, alias):
    if _decimal(alias) is not None:
        # A numeric identifier is an identifier only in an explicit subject
        # position. The same digits elsewhere can be an unsupported quantity.
        label = r'(?P<alias>' + re.escape(alias) + ')'
        explicit = re.compile(r'(?:样本|sample)\s*[:：]?\s*' + label + r'(?![A-Za-z0-9_])', re.I)
        bare = re.compile(r'(?:^|[，,；;\n])\s*' + label + r'(?=\s*[:：])', re.M)
        return [match.span('alias') for pattern in (explicit, bare) for match in pattern.finditer(text)]
    pattern = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(alias) + r'(?![A-Za-z0-9_])')
    return [match.span() for match in pattern.finditer(text)]


def _presentation_mask(answer, hits, aliases):
    value = _mask(answer, CITATION)
    value = _mask(value, r'(?m)^\s*\d+[.、)](?=\s|[^\d])\s*')
    # Source versions are metadata only when explicitly labelled and supplied.
    for hit in hits:
        version = str(hit.get('version') or '')
        if version:
            pattern = re.compile(r'(?:版本|version)\s*[:：]?\s*' + re.escape(version), re.I)
            value = _mask(value, pattern)
    for alias in sorted(aliases, key=len, reverse=True):
        for start, end in _alias_spans(value, alias):
            value = value[:start] + ' ' * (end - start) + value[end:]
    return value


def _bounds(text, start, end, breaks):
    left = max((match.end() for match in breaks.finditer(text, 0, start)), default=0)
    right_match = breaks.search(text, end)
    return left, right_match.start() if right_match else len(text)


def _sample_subject(answer, start, aliases):
    left, _ = _bounds(answer, start, start, SENTENCE_BREAK)
    before = answer[left:start]
    mentions = []
    for alias, sample in aliases.items():
        mentions.extend((start, end, sample) for start, end in _alias_spans(before, alias))
    for match in re.finditer(r'(?:样本|sample)\s*[:：]?\s*([A-Za-z0-9_][A-Za-z0-9_.-]*)', before, re.I):
        identifier = match[1].rstrip('.')
        mentions.append((match.start(1), match.end(1), aliases.get(identifier, 'unknown:' + identifier)))
    if mentions:
        ordered = sorted(set(mentions))
        latest = ordered[-1]
        earlier = [mention for mention in ordered[:-1] if mention[2] != latest[2]]
        if earlier:
            between = before[earlier[-1][1]:latest[0]]
            if re.search(r'[与和及、]|分别|合计', between) and not NUMBER.search(between):
                return None
        return latest[2]
    samples = set(aliases.values())
    return next(iter(samples)) if len(samples) == 1 else None


def _metric_and_unit(context, suffix, terms):
    unit_match = re.match(r'\s*(平方公里|平方米|公顷|km[²2]|m[²2]|像素|px(?:[²2])?|%|％|百分比|张|幅|条|个)', suffix, re.I)
    raw_unit = unit_match[1].lower() if unit_match else None
    if raw_unit in {'平方公里', '平方米', '公顷', 'km²', 'km2', 'm²', 'm2'} or re.search(r'平方米|平方公里|公顷', context):
        return 'physical_area', 'ground_area'
    unit = ({'像素': 'pixels', 'px': 'pixels', 'px²': 'pixels', 'px2': 'pixels',
             '%': 'percent', '％': 'percent', '百分比': 'percent', '张': 'samples',
             '幅': 'samples', '条': 'records'}.get(raw_unit))
    if '百分之' in context or '百分比' in context:
        unit = 'percent'
    matches = [(len(term), context.rfind(term), metric)
               for metric, names in terms.items() for term in names if term in context]
    prefix = context[:len(context) - len(suffix)] if suffix else context
    prefix_matches = [(len(term), prefix.rfind(term), metric)
                      for metric, names in terms.items() for term in names if term in prefix]
    explicit_metric = max(prefix_matches)[2] if prefix_matches else None
    if not raw_unit and re.match(r'\s*(?:[A-Za-z_][\w²]*|万|亿|千|百|亩|平方)', suffix):
        return explicit_metric, 'unsupported_unit'
    # Unit-bearing nouns identify a metric even if its value comes first.
    if re.match(r'\s*个\s*(?:对象|目标|连通域)', suffix):
        return explicit_metric or 'object_count', 'objects'
    if unit == 'pixels':
        if re.search(r'最大对象|最大连通域|平均|阈值|阈限', context):
            return None, unit
        return explicit_metric or 'changed_px', unit
    if not matches:
        return None, unit
    metric = max(matches)[2]
    if unit == 'percent':
        metric = {'changed_count': 'changed_share', 'review_count': 'review_share'}.get(metric, metric)
        if metric.startswith(('risk:', 'reason:', 'event:', 'scale:')):
            metric += '_share'
    if raw_unit == '个':
        unit = 'objects' if metric == 'object_count' else ('samples' if metric.endswith('_count') else None)
    return metric, unit


def _local_citations(answer, start, end):
    left, right = _bounds(answer, start, end, SENTENCE_BREAK)
    if left > 0 and answer[left - 1] != '\n':
        # A reference immediately after the preceding full stop belongs to it.
        leading = re.match(r'(?:\s*\[\d+\])+', answer[left:])
        if leading:
            left += leading.end()
    if right < len(answer):
        right += 1
    trailing = re.match(r'(?:\s*\[\d+\])*', answer[right:])
    if trailing:
        right += trailing.end()
    return [int(match[1]) for match in CITATION.finditer(answer[left:right])]


def _literal_source_support(clause, answer, start, end, hits):
    normalize = lambda value: re.sub(r'\s+|\[\d+\]', '', value).strip('。！？!?；;，,：:')
    claim = normalize(clause)
    if len(claim) < 4:
        return False
    return any(claim in normalize(str(hits[index - 1].get('text') or ''))
               for index in _local_citations(answer, start, end) if 1 <= index <= len(hits))


def _forbidden_absolutes(answer):
    patterns = [r'(?:所有|全部|任何).{0,30}?(?:均|都|完全|百分之百)?(?:准确|正确|无误)',
                r'(?:100\s*%|百分之百).{0,8}?(?:准确|正确)',
                r'(?:无需|不用|不必|不需要).{0,6}?(?:人工检查|人工复核|人工确认|复核|检查)']
    for pattern in patterns:
        for match in re.finditer(pattern, answer):
            prefix = answer[max(0, match.start() - 16):match.start()]
            prefix = re.split(r'[，,；;。！？!?\n]', prefix)[-1]
            if re.search(r'不能(?:认为|断言|说|保证)|不可(?:认为|断言|说)|并非|不代表|不意味着|无法保证', prefix):
                continue
            yield match[0]


def validate_claims(answer: str, hits: list[dict], facts) -> dict:
    """Check conservative numeric bindings and basic prohibited absolutes.

    ``ok`` means only that the checks listed in ``scope`` passed. Unknown or
    ambiguous numeric claims fail closed; ordinary nonnumeric narrative does
    not receive semantic entailment verification.
    """
    answer = str(answer or '')
    refs = [int(match[1]) for match in CITATION.finditer(answer)]
    citation_ok = bool(refs) and all(1 <= index <= len(hits) for index in refs)
    reasons = []
    if not citation_ok:
        reasons.append({'code': 'citation_range', 'claim': 'missing or out-of-range citation'})
    records, aliases, terms = _structured_facts(facts)
    numeric_pool = {record.value for record in records}
    numeric_pool.update(record.value * 100 for record in records if record.unit == 'ratio')
    for hit in hits:
        numeric_pool.update(_number_value(match[0]) for match in NUMBER.finditer(str(hit.get('text') or '')))
    masked = _presentation_mask(answer, hits, aliases)
    numbers = list(NUMBER.finditer(masked))
    for number in numbers:
        value = _number_value(number[0])
        left, right = _bounds(masked, number.start(), number.end(), CLAUSE_BREAK)
        # Do not let a neighbouring number's metric name bind this value.
        previous = [other for other in numbers if left <= other.start() < number.start()]
        following = [other for other in numbers if number.end() <= other.start() < right]
        context_left = previous[-1].end() if previous else left
        context_right = following[0].start() if following else right
        context = masked[context_left:context_right]
        suffix = masked[number.end():context_right]
        metric, unit = _metric_and_unit(context, suffix, terms)
        sample = _sample_subject(answer, number.start(), aliases) if metric in SAMPLE_METRICS else None
        candidates = [record for record in records if record.metric == metric and record.sample == sample]
        # Never silently interpret a comparison or threshold as an equality.
        is_comparison = bool(re.search(r'至少|至多|超过|大于|小于|不超过|低于|高于|阈值|阈限', context))
        valid = False
        for record in candidates:
            expected_unit = unit or record.unit
            expected = record.value * 100 if expected_unit == 'percent' and record.unit == 'ratio' else record.value
            same_unit = expected_unit == record.unit or (expected_unit == 'percent' and record.unit == 'ratio')
            # Percentages can use displayed decimal rounding, never integer truncation.
            if same_unit and not is_comparison:
                if expected_unit == 'percent' and expected.as_tuple().exponent < value.as_tuple().exponent:
                    expected = expected.quantize(Decimal(1).scaleb(value.as_tuple().exponent), rounding=ROUND_HALF_UP)
                if expected == value:
                    valid = True
                    break
        if valid:
            continue
        if metric not in SAMPLE_METRICS and metric != 'physical_area' and not candidates:
            if _literal_source_support(answer[left:right], answer, number.start(), number.end(), hits):
                continue
        if value not in numeric_pool:
            code = 'unsupported_numeric_value'
        elif metric == 'physical_area' or (candidates and unit and all(
                unit != record.unit and not (unit == 'percent' and record.unit == 'ratio') for record in candidates)):
            code = 'unit_mismatch'
        elif metric in SAMPLE_METRICS and sample is None:
            code = 'ambiguous_sample'
        elif metric is None or not candidates or is_comparison:
            code = 'unbound_numeric_claim'
        else:
            code = 'sample_metric_mismatch'
        reasons.append({'code': code, 'claim': answer[left:right].strip(), 'value': str(value),
                        'sample': sample, 'metric': metric, 'unit': unit})
    for claim in _forbidden_absolutes(answer):
        reasons.append({'code': 'forbidden_absolute', 'claim': claim})
    numeric_ok = not any(reason['code'] not in {'citation_range', 'forbidden_absolute'} for reason in reasons)
    return {'ok': not reasons, 'reasons': reasons,
            'scope': {**validation_scope(), 'checkedNumericClaims': len(numbers)},
            'citationRange': citation_ok, 'numericClaims': numeric_ok,
            'basicAbsolutes': not any(reason['code'] == 'forbidden_absolute' for reason in reasons)}
