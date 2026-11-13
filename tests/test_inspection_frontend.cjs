const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const file = path.join(root, 'web/inspection_evidence.js');
assert.ok(fs.existsSync(file), 'Missing scope-aware evidence presentation');
const evidence = require(file);

let legacyCalls = 0;
const legacy = () => { legacyCalls++; return 'WRONG CURRENT SAMPLE'; };
assert.equal(evidence.guardAnswer({answer:'连通域是一组相连像素。[1]', answerScope:'knowledge'}, {id:'A'}, legacy), '连通域是一组相连像素。[1]');
assert.equal(legacyCalls, 0, 'Knowledge answer must bypass sample-specific checks');

let target;
const sampleAnswer = evidence.guardAnswer({answer:'B：900px，对象4个，中部。', answerScope:'sample', evidenceSamples:[{sample_id:'B',changed_px:900,object_count:4,grid_position:'中部'}]}, {id:'A'}, item => {target=item; return 'checked';});
assert.equal(sampleAnswer,'checked');
assert.equal(target.id,'B','Followup must validate returned sample instead of current selection');

const ranked = {answer:'1. B：900px，对象4个。\n2. C：800px，对象3个。',answerScope:'list',evidenceSamples:[{sample_id:'B',changed_px:900,object_count:4},{sample_id:'C',changed_px:800,object_count:3}]};
assert.equal(evidence.guardAnswer(ranked,{id:'A'},legacy),ranked.answer);
const multiline={answer:'第2张是 B，变化像素900px。\n\n变化像素900px，对象数量4个，中部。\n[1] 通用说明',answerScope:'mixed',evidenceSamples:[{sample_id:'B',changed_px:900,object_count:4}]};
assert.equal(evidence.guardAnswer(multiline,{id:'A'},legacy),multiline.answer,'Multiline sample facts must not trigger false warnings');
assert.deepEqual(evidence.samplePicks(ranked,[{id:'A',stem:'A'},{id:'B',stem:'B'},{id:'C',stem:'C'}]).map(x=>x.id),['B','C']);
assert.match(evidence.guardAnswer({...ranked,answer:'1. B：999px，对象4个。\n2. C：800px，对象3个。'},{id:'A'},legacy),/核对/);

const unsafe = evidence.citationsHtml([{id:1,title:'<img src=x onerror=alert(1)>',section:'检查',excerpt:'<script>bad()</script>',source:'rules.md',version:'v1'}]);
assert.ok(!unsafe.includes('<img'));
assert.ok(!unsafe.includes('<script>'));
assert.match(unsafe,/&lt;script&gt;/);

const generated = {status:'done',text:'生成的统一摘要[1]',citations:[{id:1,title:'复核说明',section:'碎片化',source:'quality.md',version:'v1',excerpt:'检查小连通域'}],snapshotFingerprint:'snapshot-1'};
assert.equal(evidence.reportSummary(generated,'snapshot-1','本地摘要').text,'生成的统一摘要[1]');
assert.equal(evidence.reportSummary(generated,'snapshot-2','本地摘要').text,'本地摘要');

// Execute the real report serializer. Only page-state and formatting dependencies are supplied.
const html = fs.readFileSync(path.join(root,'web/index.html'),'utf8');
const start = html.indexOf('function reportPlainText('), end = html.indexOf('function downloadRegionReportWord(',start);
const load = new Function('InspectionEvidence','reportRecord','reportSnapshotFingerprint','buildLocalReportSummary','state','fmt','reportPercent', html.slice(start,end)+';return reportPlainText;');
const serialize = load(evidence,()=>({summary:generated}),()=> 'snapshot-1',()=> '本地摘要',{classFilter:'all'},String,(a,b)=> String(a/b));
const snapshot = {key:'x',region:{short:'测试'},filterText:'全部',totalCount:4,changedCount:4,reviewCount:2,scaleBuckets:[],eventBuckets:[],riskBuckets:[],reasonBuckets:[],representatives:[],dominantScale:'局部变化'};
const text = serialize(snapshot);
assert.match(text,/生成的统一摘要/);
assert.match(text,/复核说明/);
assert.ok(!text.includes('本地摘要'),'Export must use current generated summary');
const compactStart=html.indexOf('function compactReportSnapshot('),compactEnd=html.indexOf('async function callReportSummaryApi(',compactStart);
const compact = new Function(html.slice(compactStart,compactEnd)+';return compactReportSnapshot;')();
const packed=compact({...snapshot,region:{id:'test'},riskBuckets:[{key:'high',label:'高风险复核',count:2,ratio:.5}],correctionEntries:[{},{}]});
assert.equal(packed.riskBuckets[0].key,'high','Report API must receive risk keys to avoid resetting high-risk counts to zero');
assert.equal(packed.correctionCount,2,'Report API must receive manual correction count');
const fingerprintStart=html.lastIndexOf('function reportSnapshotFingerprint('),fingerprintEnd=html.indexOf('function syncLocalReportRecord(',fingerprintStart);
const fingerprint=new Function('compactReportSnapshot',html.slice(fingerprintStart,fingerprintEnd)+';return reportSnapshotFingerprint;')(compact);
const baseSnapshot={...snapshot,region:{id:'test'},correctionEntries:[]};
assert.notEqual(fingerprint(baseSnapshot),fingerprint({...baseSnapshot,eventBuckets:[{key:'a',label:'A',count:2}]}),'Event distribution changes must invalidate summary');
assert.notEqual(fingerprint(baseSnapshot),fingerprint({...baseSnapshot,scaleBuckets:[{key:'small',label:'小区域',count:2}]}),'Scale distribution changes must invalidate summary');
// Execute the actual downloader and inspect the Blob it hands to the browser.
const downloadStart=html.indexOf('function downloadRegionReportWord('),downloadEnd=html.indexOf('function reportRecord(',downloadStart);
let capturedBlob;
const fakeDocument={createElement:()=>({click(){}})};
const fakeURL={createObjectURL(blob){capturedBlob=blob;return 'blob:test';},revokeObjectURL(){}};
const makeDownload=new Function('activeRegion','buildRegionReportSnapshot','reportPlainText','esc','state','Blob','URL','document','fmt','reportPercent',html.slice(downloadStart,downloadEnd)+';return downloadRegionReportWord;');
const download=makeDownload(()=>snapshot.region,()=>snapshot,serialize,x=>String(x).replace(/</g,'&lt;'),{classFilter:'all'},Blob,fakeURL,fakeDocument,String,(a,b)=>String(a/b));
download();
capturedBlob.text().then(content=>{assert.match(content,/生成的统一摘要/);assert.match(content,/复核说明/);console.log('Frontend scope, source escaping, snapshot metadata, and actual downloaded report tests passed.');});
