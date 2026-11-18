"""Native MathML for the Transformer lesson, shared by reading and playback."""
import re
from html import escape


def tag(name, *children):
    return f'<{name}>' + ''.join(children) + f'</{name}>'


def mi(value):
    return tag('mi', escape(value))


def mo(value):
    return tag('mo', escape(value))


def mn(value):
    return tag('mn', str(value))


def row(*children):
    return tag('mrow', *children)


def sub(name, index):
    return tag('msub', mi(name), mn(index) if isinstance(index, int) else mi(index))


def sup(base, exponent):
    return tag('msup', base, exponent)


def par(*parts):
    return row(mo('('), *parts, mo(')'))


def call(name, *args):
    return row(mi(name), mo('⁡'), par(*args))


def equals(lhs, *rhs):
    return row(lhs, mo('='), *rhs)


def frac(top, bottom):
    return tag('mfrac', top, bottom)


def sqrt(*parts):
    return tag('msqrt', *parts)


def sum_from(index, upper):
    lower = equals(mi(index), mn(1))
    return tag('munderover', mo('∑'), lower, upper)


def sum_at(index):
    return tag('munder', mo('∑'), mi(index))


x, h, y, q, k, V, Q, K = map(mi, ('x', 'h', 'y', 'q', 'k', 'V', 'Q', 'K'))
x1, h2, dk = sub('x', 1), sub('h', 2), sub('d', 'k')
indexed_x = sub('x', 'j')
delta_a, delta_m = mi('ΔA'), mi('ΔM')
h_final = sub('h', 'final')
rms = sqrt(frac(row(sum_from('j', mi('d')), sup(indexed_x, mn(2))), mi('d')), mo('+'), mi('ε'))


def shifted_score(index):
    return row(sub('s', index), mo('−'), mi('m'))


def normalized_exp(index, sum_index='j'):
    return frac(
        sup(mi('e'), shifted_score(index)),
        row(sum_at(sum_index), sup(mi('e'), shifted_score(sum_index))),
    )


def prime(symbol):
    return sup(symbol, mo('′'))


def attention_score(query, key, scale):
    return row(frac(row(query, sup(key, mo('⊤'))), sqrt(scale)), mo('+'), mi('M'))


def matrix_shape(name, rows, cols):
    return row(name, mo('∈'), sup(mi('ℝ'), row(rows, mo('×'), cols)))


def vector(*entries):
    return row(mo('['), *[part for i, n in enumerate(entries)
                           for part in ((mo(','), mn(n)) if i else (mn(n),))], mo(']'))


def layer_at(index):
    return sup(x, par(index))


def rotation(symbol, position):
    return equals(prime(mi(symbol)), sub('R', position), mi(symbol))


def projections(branch, weight, inp=h2):
    return equals(mi(branch), inp, sub('W', weight))


EQUATIONS = {
    'embedding': (equals(sub('x', 't'), mi('E'), mo('['), sub('id', 't'), mo(']')),),
    'embedding_shape': (matrix_shape(mi('E'), sub('N', 'vocab'), mi('d')), matrix_shape(mi('X'), mi('T'), mi('d'))),
    'embedding_demo': (equals(mi('E'), mo('['), mn(17), mo(']'), mo('='), vector('0.2', '0.3')),),
    'rms': (equals(mi('r'), rms), equals(sub('h','i'), sub('γ', 'i'), frac(sub('x', 'i'), mi('r')))),
    'rms_example': (row(mi('r'), mo('='), sqrt(frac(row(sup(mn('0.2'), mn(2)), mo('+'), sup(mn('0.3'), mn(2))), mn(2))),
                         mo('≈'), mn('0.255')),),
    'rms_demo': (equals(mi('r'), sqrt(call('mean', sup(x, mn(2))))),),
    'rms_divide': (equals(mi('n'), frac(x, mi('r'))),),
    'rms_gain': (equals(h, mi('n'), mo('⊙'), mi('γ')),),
    'qkv': tuple(equals(target, mi('H'), sub('W', name))
                 for target, name in ((Q, 'Q'), (K, 'K'), (V, 'V'))),
    'score': (equals(mi('S'), attention_score(prime(Q), prime(K), dk)),),
    'score_demo': (equals(mi('S'), attention_score(prime(Q), prime(K), mn(2))),),
    'score_cell': (equals(sub('s', 'ij'), frac(row(prime(sub('q', 'i')), mo('·'), prime(sub('k', 'j'))), sqrt(dk)),
                          mo('+'), sub('M', 'ij')),),
    'mask': (equals(sub('M', 'ij'), mo('{'), tag('mtable',
              tag('mtr', tag('mtd', mn(0)), tag('mtd', row(mi('j'), mo('≤'), mi('i')))),
              tag('mtr', tag('mtd', row(mo('−'), mo('∞'))), tag('mtd', row(mi('j'), mo('>'), mi('i')))))),),
    'score_lab': (equals(sub('s','i'), frac(row(Q, mo('·'), sub('K','i')), sqrt(mn(2)))),),
    'weights': (equals(sub('α', 'ij'), frac(sup(mi('e'), sub('s', 'ij')), row(sum_at('r'), sup(mi('e'), sub('s', 'ir'))))),
                equals(sub('o', 'i'), sum_at('j'), sub('α', 'ij'), sub('v', 'j'))),
    'attention_mix': (equals(mi('o'), sum_at('j'), sub('α','j'), sub('V','j')),),
    'attention_contribution': (row(mn('0.252'), mo('·'), par(mn('−0.393')), mo('≈'), mn('−0.099')),),
    'attention_mix_lab': (equals(mi('o'), sum_at('j'), sub('w','j'), sub('V','j')),),
    'softmax_attention': (equals(mi('m'), call('max', mi('s'))), equals(sub('α', 'i'), normalized_exp('i'))),
    'softmax_vocab': (equals(mi('m'), call('max', mi('s'))), equals(sub('p', 'c'), normalized_exp('c', sum_index='b'))),
    'softmax_lab_attention': (equals(mi('m'), call('max', mi('s'))), equals(sub('w', 'i'), normalized_exp('i'))),
    'softmax_ratio': (equals(frac(sub('α','i'), sub('α','j')), sup(mi('e'), row(sub('s','i'), mo('−'), sub('s','j')))),),
    'softmax_gradient': (equals(frac(row(mo('∂'),sub('α','i')), row(mo('∂'),sub('s','i'))), sub('α','i'), par(mn(1),mo('−'),sub('α','i'))),),
    'dot_spread': (
        equals(tag('msub',mi('s'),tag('mtext','raw')), q,mo('·'),k,mo('='),sum_from('i',dk),sub('q','i'),sub('k','i')),
        equals(call('Var',sub('s','raw')),dk),
        equals(call('Std',sub('s','raw')),sqrt(dk)),
    ),
    'scale_compare': (
        equals(call('Std',frac(sub('s','raw'),sqrt(dk))),mn(1)),
        equals(call('Std',frac(sub('s','raw'),dk)),frac(mn(1),sqrt(dk))),
    ),
    'rope': (rotation('q','p'),rotation('k','r')),
    'rope_demo': (
        equals(sup(q,mo('′')),call('R',mi('p'),mi('θ')),q),
        equals(sup(k,mo('′')),call('R',mi('r'),mi('θ')),k),
    ),
    'rope_score': (equals(mi('s'),frac(row(sup(q,mo('′')),mo('·'),sup(k,mo('′'))),sqrt(mn(2)))),),
    'rope_relative': (
        equals(row(prime(q),mo('·'),prime(k)),sup(q,mo('⊤')),
               tag('msub', mi('R'), row(mi('r'), mo('−'), mi('p'))),k),
    ),
    'head_attention': (equals(sub('head','a'),call('softmax', attention_score(prime(sub('Q','a')), prime(sub('K','a')), dk)),
                              sub('V','a')),),
    'concat_heads': (equals(mi('C'), call('Concat',sub('head',1),mo(','),sub('head',2))),),
    'attention_projection': (equals(delta_a,call('Concat',sub('head',1),mo(','),mo('…'),mo(','),sub('head','n')),sub('W','O')),),
    'attention_shape': (matrix_shape(sub('W','O'), row(mi('n'), mo('·'), sub('d','v')), mi('d')),),
    'residual_component': (equals(sub('x', '1,i'), sub('x', 'i'), mo('+'), sub('ΔA', 'i')),),
    'residual_slope': (equals(frac(row(mo('∂'), sub('x', '1,i')), row(mo('∂'), sub('x','i'))),
                              mn(1), mo('+'), frac(row(mo('∂'),sub('ΔA','i')), row(mo('∂'),sub('x','i')))),),
    'residual_a': (equals(x1,x,mo('+'),delta_a),),
    'residual_f': (equals(y,x1,mo('+'),delta_m),),
    'pre_norm_layer': (
        equals(sub('X',1), mi('X'), mo('+'), call('Attention', call('RMSNorm₁', mi('X')))),
        equals(mi('Y'), sub('X',1), mo('+'), call('FFN', call('RMSNorm₂', sub('X',1)))),
    ),
    'norm_2': (equals(h2,call('RMSNorm',x1)),),
    'ffn_project': (projections('u','up'),projections('g','gate')),
    'silu': (equals(sub('a','j'),call('SiLU',sub('g','j')),mo('='),sub('g','j'),mo('·'),call('σ',sub('g','j'))),),
    'sigmoid': (equals(call('σ',mi('t')),frac(mn(1),row(mn(1),mo('+'),sup(mi('e'),row(mo('−'),mi('t')))))),),
    'gate': (equals(mi('z'),mi('a'),mo('⊙'),mi('u')),),
    'ffn_down': (equals(delta_m,mi('z'),sub('W','down')),),
    'ffn_units': (equals(sub('u','j'), sum_from('i', mi('d')), sub('h','2,i'), sub('W','up,ij')),
                  equals(sub('z','j'), sub('a','j'), mo('·'), sub('u','j'))),
    'ffn_down_sum': (equals(sub('ΔM','i'), sum_from('j', mi('m')), sub('z','j'), sub('W','down,ji')),),
    'ffn_down_example': (row(mn('−0.079'), mo('·'), vector('0.1', '−0.5'), mo('≈'), vector('−0.008', '0.040')),),
    'layer_flow': (
        equals(layer_at(row(mi('ℓ'), mo('+'), mn(1))),
               tag('msub', mi('Layer'), row(mi('ℓ'), mo('+'), mn(1))), par(layer_at(mi('ℓ')))),
        equals(h_final,call('FinalNorm',layer_at(mi('N')))),
    ),
    'layer_2': (equals(layer_at(mn(2)), sub('Layer',2), par(layer_at(mn(1)))),),
    'layer_3': (equals(layer_at(mn(3)), sub('Layer',3), par(layer_at(mn(2)))),),
    'lm_head': (equals(mi('s'),h_final,sub('W','vocab')), matrix_shape(sub('W','vocab'), mi('d'), sub('N','vocab'))),
    'logit': (equals(sub('s','c'),h_final,mo('·'),sub('w','c')),),
    'logit_example': (row(vector('1.6812','0.1542'), mo('·'), vector('0.9','0.6'), mo('≈'), mn('1.606')),),
    'logit_margin': (equals(row(sub('s','a'), mo('−'), sub('s','b')), h_final, mo('·'), par(sub('w','a'), mo('−'), sub('w','b'))),),
    'lab_logit': (equals(sub('s','c'),y,mo('·'),sub('w','c')),),
    'greedy': (equals(sub('c','next'),row(tag('munder',mi('argmax'),mi('c')),sub('s','c'))),),
    'sampling': (equals(sub('F','c'), tag('munder', mo('∑'),row(mi('b'),mo('≤'),mi('c'))), sub('p','b')),
                 equals(sub('c','next'), call('min', mo('{'), mi('c'),mo('∣'),sub('F','c'),mo('≥'),mi('u'),mo('}')))),
    'rope_compare': (
        row(call('cos',row(mn(45),mo('°'))),mo('≈'),mn('0.707')),
        equals(call('cos',row(mn(90),mo('°'))),mn(0)),
    ),
    'rms_lab': (equals(mi('r'), sqrt(call('mean', sup(x1,mn(2))))), equals(h, frac(x1,mi('r')))),
    'lab_residual_a': (equals(x1, sub('x',6), mo('+'), mi('o')),),
    'ablation_renormalize': (equals(prime(sub('w','j')),frac(sub('w','j'),row(mn(1),mo('−'),sub('w',1)))),),
    'ffn_lab': (projections('g','g',h), projections('u','u',h), equals(mi('a'),call('SiLU',mi('g'))),
                equals(mi('z'),mi('a'),mo('⊙'),mi('u')), equals(delta_m,sum_at('j'),sub('z','j'),sub('W','d,j'))),
}

PLAYBACK = (
    'embedding', 'rms_demo', 'rms_divide', 'rms_gain', 'score_demo', 'softmax_attention',
    'attention_mix', 'head_attention', 'concat_heads', 'attention_projection', 'rope_demo',
    'rope_score', 'residual_a', 'residual_f', 'ffn_project', 'silu', 'gate', 'ffn_down',
    'layer_2', 'layer_3', 'logit', 'softmax_vocab', 'greedy', 'sampling',
    'score_lab', 'ffn_lab', 'lab_logit', 'softmax_lab_attention', 'attention_mix_lab', 'rms_lab', 'lab_residual_a',
)


def math_html(name):
    return ''.join(
        f'<math display="block">{expression}</math>' for expression in EQUATIONS[name]
    )


def render_equations(source):
    return re.sub(r'\{\{math:([a-z0-9_]+)\}\}', lambda match: math_html(match[1]), source)


def playback_templates():
    return ''.join(f'<template id="equation-{name}">{math_html(name)}</template>' for name in PLAYBACK)
