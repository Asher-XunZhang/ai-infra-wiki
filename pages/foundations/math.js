/* Native MathML for live numeric readouts in the foundation lessons. */
(() => {
 'use strict';
 const escape = value => String(value).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
 const tag = (name, ...children) => `<${name}>${children.join('')}</${name}>`;
 const token = name => value => tag(name, escape(value));
 const num = token('mn'), op = token('mo'), variable = token('mi'), text = token('mtext');
 const row = (...items) => tag('mrow', ...items);
 const join = (items, operator) => items.join(op(operator));
 const array = (...items) => row(op('['), join(items, ','), op(']'));
 const vector = (values, format = String) => array(...values.map(x => num(format(x))));
 const matrix = values => {
  const rows = values.map(line => tag('mtr', ...line.map(value => tag('mtd', num(value)))));
  return row(op('['), tag('mtable', ...rows), op(']'));
 };
 const display = (...formulas) => `<div class="foundation-math">${formulas.map(
  parts => `<math display="block">${parts}</math>`).join('')}</div>`;
 window.FoundationMath = {
  num, op, variable, text, row, join, vector, matrix, display,
  sub: (name, index) => tag('msub', variable(name), Number.isInteger(index) ? num(index) : text(index)),
  frac: (top, bottom) => tag('mfrac', top, bottom),
  sqrt: value => tag('msqrt', value),
 };
})();
