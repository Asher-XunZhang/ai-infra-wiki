const assert=require('node:assert/strict');
const S=require('../../pages/motion/generation-streaming.js');

const ready=S.state('learning',0);
assert.equal(ready.selected,false);
assert.equal(ready.clientText,'');
assert.equal(ready.stage,'ready');

const selected=S.state('learning',1.5);
assert.equal(selected.selected,true);
assert.equal(selected.detokenized,false);
assert.equal(selected.clientText,'');
assert.equal(selected.nextInputId,101);
assert.equal(selected.stage,'selected');

const detok=S.state('learning',2.5);
assert.equal(detok.detokenized,true);
assert.equal(detok.inFlight,false);
assert.equal(detok.clientText,'');
assert.equal(detok.stage,'detokenized');

const flight=S.state('learning',3.5);
assert.equal(flight.inFlight,true);
assert.equal(flight.clientText,'');
assert.equal(flight.nextInputId,101);
assert.deepEqual(flight.candidates.map(x=>x.word),['句号','其他']);
assert.equal(flight.stage,'in-flight');
assert.ok(flight.packetProgress>0&&flight.packetProgress<1);

const received=S.state('learning',4);
assert.equal(received.received,true);
assert.equal(received.clientText,'学习');
assert.equal(received.nextInputId,received.id);
assert.equal(received.stage,'received');

const rest=S.state('rest',4);
assert.equal(rest.clientText,'休息');
assert.equal(rest.nextInputId,102);
assert.deepEqual(rest.candidates.map(x=>x.word),['一下','其他']);
assert.notEqual(rest.id,received.id);
assert.equal(S.state('rest',3.5).clientText,'','client still has no text while the packet is in flight');

console.log('PASS: streaming states keep selected ID, detokenized fragment, in-flight packet and client receipt separate; model input follows the selected ID before receipt.');
