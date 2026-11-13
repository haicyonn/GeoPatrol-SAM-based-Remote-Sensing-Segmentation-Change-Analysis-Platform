// Evaluate the real generated legacy script, rather than inspecting only text.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const folder=process.argv[2];
const html=fs.readFileSync(path.join(folder,'index.html'),'utf8');
const inline=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');
let renders=0,downloadName='',blobValue=null;
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{id,dataset:{},classList:{add(){},remove(){},contains(){return false;},toggle(){}},style:{},innerHTML:'',value:'',onclick:null,click(){downloadName=this.download||'';}});return nodes.get(id);}
const sandbox={console,TextDecoder,Uint8Array,document:{getElementById:node,querySelectorAll:()=>[],addEventListener(){},createElement:()=>node('download')},
  localStorage:{getItem:()=>null,setItem(){},removeItem(){}},sessionStorage:{getItem:()=>null,setItem(){}},
  location:{origin:'http://example.test',pathname:'/index.html',search:'',hash:''},
  URL:{createObjectURL(value){blobValue=value;return 'blob:test';},revokeObjectURL(){}},
  Blob,fetch(){throw Error('Legacy code must not fetch before authorization');},setTimeout(){throw Error('Legacy bootstrap must be deferred');},clearTimeout(){},
  requestAnimationFrame(){throw Error('Legacy animation must not bootstrap automatically');},navigator:{},addEventListener(){},history:{replaceState(){}},
};sandbox.window=sandbox;sandbox.globalThis=sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(folder,'inspection_evidence.js'),'utf8'),sandbox);
vm.runInContext(inline,sandbox);
assert.equal(sandbox.ConsoleLegacy.app.regions.length,0);
sandbox.ConsoleLegacy.install({render(){renders++;}});
assert.equal(renders,0);
for(const [ratio,width] of [[0,0],[0.2,20],[1,100]]){
  const meter=sandbox.ConsoleLegacy.reportMeter([{label:'bucket',count:Math.round(ratio*10),ratio}],10,'empty');
  assert.match(meter,new RegExp('width:'+width+'%'));
}
sandbox.ConsoleLegacy.setContext({app:{regions:[{id:'whu',short:'WHU',items:[]}]}});
sandbox.ConsoleLegacy.setWordProvider({snapshot:()=>({filterText:'全部样本'}),text:()=>'WHU\n受控摘要\n<script>private</script>',scope:()=>'全部样本'});
sandbox.ConsoleLegacy.downloadWord();
assert.match(downloadName,/\.doc$/);
blobValue.text().then(text=>{
  assert.match(text,/&lt;script&gt;private&lt;\/script&gt;/);
  assert.doesNotMatch(text,/<script>private/);
  sandbox.ConsoleLegacy.clear();assert.equal(sandbox.ConsoleLegacy.app.regions.length,0);
  console.log('Generated empty legacy runtime and original Word exporter verified');
}).catch(error=>{console.error(error);process.exitCode=1;});
