const API = '';
let f1Data = null, demoSentences = [], currentMetric = 'f1', f1Chart = null;
const entityColors = {
  'PER':'#a864dc','I-PER':'#a864dc','B-PER':'#a864dc','E-PER':'#a864dc','S-PER':'#a864dc','per':'#a864dc','I-per':'#a864dc','B-per':'#a864dc','E-per':'#a864dc','S-per':'#a864dc',
  'GEO':'#3498db','I-GEO':'#3498db','B-GEO':'#3498db','E-GEO':'#3498db','S-GEO':'#3498db','geo':'#3498db','I-geo':'#3498db','B-geo':'#3498db','E-geo':'#3498db','S-geo':'#3498db',
  'ORG':'#f1943f','I-ORG':'#f1943f','B-ORG':'#f1943f','E-ORG':'#f1943f','S-ORG':'#f1943f','org':'#f1943f','I-org':'#f1943f','B-org':'#f1943f','E-org':'#f1943f','S-org':'#f1943f',
  'GPE':'#27ae60','I-GPE':'#27ae60','B-GPE':'#27ae60','E-GPE':'#27ae60','S-GPE':'#27ae60','gpe':'#27ae60','I-gpe':'#27ae60','B-gpe':'#27ae60','E-gpe':'#27ae60','S-gpe':'#27ae60',
  'TIM':'#f1c40f','I-TIM':'#f1c40f','B-TIM':'#f1c40f','E-TIM':'#f1c40f','S-TIM':'#f1c40f','tim':'#f1c40f','I-tim':'#f1c40f','B-tim':'#f1c40f','E-tim':'#f1c40f','S-tim':'#f1c40f',
  'ART':'#e84393','I-ART':'#e84393','B-ART':'#e84393','E-ART':'#e84393','S-ART':'#e84393','art':'#e84393','I-art':'#e84393','B-art':'#e84393','E-art':'#e84393','S-art':'#e84393',
  'EVE':'#00cec9','I-EVE':'#00cec9','B-EVE':'#00cec9','E-EVE':'#00cec9','S-EVE':'#00cec9','eve':'#00cec9','I-eve':'#00cec9','B-eve':'#00cec9','E-eve':'#00cec9','S-eve':'#00cec9',
  'NAT':'#6c5ce7','I-NAT':'#6c5ce7','B-NAT':'#6c5ce7','E-NAT':'#6c5ce7','S-NAT':'#6c5ce7','nat':'#6c5ce7','I-nat':'#6c5ce7','B-nat':'#6c5ce7','E-nat':'#6c5ce7','S-nat':'#6c5ce7',
};
const entityNames = {'PER':'Người','GEO':'Địa lý','ORG':'Tổ chức','GPE':'Địa-CT','TIM':'Thời gian','ART':'Tác phẩm','EVE':'Sự kiện','NAT':'Tự nhiên',
  'per':'Người','geo':'Địa lý','org':'Tổ chức','gpe':'Địa-CT','tim':'Thời gian','art':'Tác phẩm','eve':'Sự kiện','nat':'Tự nhiên'};

function getBaseType(tag) { return tag === 'O' ? null : tag.replace(/^[BIES]-/, ''); }
function getColor(tag) { return entityColors[tag] || (tag !== 'O' ? '#95a5a6' : null); }
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Particles ────────────────────────────────────────
function initParticles() {
  const c = document.getElementById('particles');
  if (!c) return;
  const colors = ['#7c8ff8','#a864dc','#3498db','#27ae60','#f1943f'];
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = 3 + Math.random() * 6;
    p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random()*100}%;top:${Math.random()*100}%;background:${colors[i%5]};animation-delay:${Math.random()*20}s;animation-duration:${15+Math.random()*15}s;`;
    c.appendChild(p);
  }
}

// ── Navigation ───────────────────────────────────────
document.querySelectorAll('.nav-item[data-page]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const pg = document.getElementById('page-' + btn.dataset.page);
    if (pg) pg.classList.add('active');
    if (btn.dataset.page === 'f1' && !f1Data) loadF1();
    if (btn.dataset.page === 'boundary') animateBars();
    if (btn.dataset.page === 'training') loadTrainingHistory();
    if (btn.dataset.page === 'ablation') loadAblation();
  });
});

// ── Status ───────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch(API + '/api/status');
    const d = await r.json();
    const dot = document.getElementById('dot-status');
    const badge = document.getElementById('model-badge');
    if (d.models_loaded.crf && d.models_loaded.softmax) {
      dot.className = 'status-dot ok'; badge.textContent = '✓ Cả hai model đã load'; badge.style.color = '#2ecc71';
    } else if (d.models_loaded.crf || d.models_loaded.softmax) {
      dot.className = 'status-dot err'; badge.textContent = '⚠ Chỉ 1 model'; badge.style.color = '#f1943f';
    } else {
      dot.className = 'status-dot err'; badge.textContent = '⚠ Demo mode (chưa train)'; badge.style.color = '#f1943f';
    }
  } catch(e) { document.getElementById('model-badge').textContent = '⚠ Không kết nối API'; }
}

// ── Demo sentences ───────────────────────────────────
async function loadDemoSentences() {
  try { const r = await fetch(API+'/api/demo-sentences'); demoSentences = await r.json(); } catch(e) {
    demoSentences = ["Barack Obama was born in Hawaii.","Apple Inc. was founded by Steve Jobs.","The United Nations met in New York on Tuesday."];
  }
  const chips = document.getElementById('demo-chips');
  demoSentences.forEach(s => {
    const c = document.createElement('button');
    c.className = 'demo-chip'; c.textContent = s.length > 55 ? s.substring(0,52)+'...' : s;
    c.title = s; c.addEventListener('click', () => { document.getElementById('input-text').value = s; doAnalyze(); });
    chips.appendChild(c);
  });
}

// ── Analyze ──────────────────────────────────────────
async function doAnalyze() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) return;
  const btn = document.getElementById('btn-analyze'), btnText = document.getElementById('btn-text');
  btn.disabled = true; btnText.innerHTML = '<span class="spinner"></span>Đang phân tích...';
  document.getElementById('crf-output').innerHTML = '<div class="loading-row"><span class="spinner"></span>CRF...</div>';
  document.getElementById('softmax-output').innerHTML = '<div class="loading-row"><span class="spinner"></span>Softmax...</div>';
  try {
    const r = await fetch(API+'/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const data = await r.json();
    renderResults(data.crf, data.softmax);
    document.getElementById('legend-box').style.display = 'flex';
    document.getElementById('comparison-row').style.display = 'block';
  } catch(e) {
    document.getElementById('crf-output').innerHTML = '<div style="color:#e74c3c;padding:12px">❌ '+e.message+'</div>';
    document.getElementById('softmax-output').innerHTML = document.getElementById('crf-output').innerHTML;
  } finally { btn.disabled = false; btnText.innerHTML = '🔍 Phân tích'; }
}

function renderResults(crf, softmax) {
  document.getElementById('crf-output').innerHTML = buildTaggedHTML(crf,false) + buildEntityList(crf);
  document.getElementById('softmax-output').innerHTML = buildTaggedHTML(softmax,true) + buildEntityList(softmax);
  buildTokenTable(crf, softmax);
}

function buildTaggedHTML(tokens, isSM) {
  if (!tokens||!tokens.length) return '<div class="placeholder">Không có kết quả</div>';
  let h = '<div class="tagged-text fadeIn">';
  for (let i=0;i<tokens.length;i++) {
    const {word,tag} = tokens[i];
    if (tag==='O') { h+=`<span class="ent ent-O">${escHtml(word)}</span>`; }
    else {
      const base = getBaseType(tag);
      const isErr = isSM && (tag.startsWith('I-') || tag.startsWith('E-')) && (i===0 || tokens[i-1].tag==='O' || tokens[i-1].tag.startsWith('S-') || tokens[i-1].tag.startsWith('E-') || getBaseType(tokens[i-1].tag)!==base);
      const cls = isErr ? 'ent-error' : 'ent-'+tag;
      h+=`<span class="ent ${cls}">${escHtml(word)}<sup class="tag-label">${isErr?tag+' ⚠':tag}</sup></span>`;
    }
  }
  return h+'</div>';
}

function buildEntityList(tokens) {
  if (!tokens) return '';
  const ents = {}; let cur=null, curT=null;
  for (const {word,tag} of tokens) {
    if (tag==='O'){cur=null;curT=null;continue;} const t=getBaseType(tag);
    if (tag.startsWith('B-') || tag.startsWith('S-')){cur=word;curT=t;} else if((tag.startsWith('I-') || tag.startsWith('E-')) && curT===t){cur+=' '+word;} else{cur=word;curT=t;}
    if(curT){if(!ents[curT])ents[curT]=new Set();ents[curT].add(cur);}
  }
  if (!Object.keys(ents).length) return '';
  let h='<div class="entity-list">';
  for (const [t,ws] of Object.entries(ents)) {
    const col=entityColors['B-'+t]||entityColors[t]||'#95a5a6';
    h+=`<div class="entity-item"><span class="entity-dot" style="background:${col}"></span><span style="color:var(--text3);width:70px;flex-shrink:0">${entityNames[t]||t}:</span><span style="color:var(--text2)">${[...ws].join(', ')}</span></div>`;
  }
  return h+'</div>';
}

function buildTokenTable(crf, softmax) {
  if (!crf||!softmax) return;
  const n = Math.min(crf.length,softmax.length,30);
  let h='<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">';
  h+='<tr><th style="padding:6px 10px;background:var(--bg2);color:var(--text3);border:1px solid var(--border);text-align:left">Từ</th>';
  h+='<th style="padding:6px 10px;background:var(--bg2);color:#f1943f;border:1px solid var(--border)">Softmax</th>';
  h+='<th style="padding:6px 10px;background:var(--bg2);color:var(--accent);border:1px solid var(--border)">CRF</th>';
  h+='<th style="padding:6px 10px;background:var(--bg2);color:var(--text3);border:1px solid var(--border)">Khớp?</th></tr>';
  for (let i=0;i<n;i++){
    const sm=softmax[i],cr=crf[i],match=sm.tag===cr.tag;
    const isE=(sm.tag.startsWith('I-') || sm.tag.startsWith('E-'))&&(i===0||softmax[i-1].tag==='O'||softmax[i-1].tag.startsWith('S-')||softmax[i-1].tag.startsWith('E-')||getBaseType(softmax[i-1].tag)!==getBaseType(sm.tag));
    h+=`<tr><td style="padding:4px 10px;border:1px solid var(--border);color:var(--text2)">${escHtml(sm.word)}</td>
    <td style="padding:4px 10px;border:1px solid var(--border);text-align:center;color:${getColor(sm.tag)||'var(--text2)'};${isE?'background:rgba(231,76,60,0.08)':''}">${sm.tag}${isE?' ⚠':''}</td>
    <td style="padding:4px 10px;border:1px solid var(--border);text-align:center;color:${getColor(cr.tag)||'var(--text2)'}">${cr.tag}</td>
    <td style="padding:4px 10px;border:1px solid var(--border);text-align:center">${match?'<span style="color:#2ecc71">✓</span>':'<span style="color:#e74c3c">✗</span>'}</td></tr>`;
  }
  document.getElementById('token-table').innerHTML = h+'</table></div>';
}

// ── F1 Chart ─────────────────────────────────────────
async function loadF1() {
  try { const r=await fetch(API+'/api/f1-scores'); f1Data=await r.json(); buildF1Chart('f1'); buildF1Table(); } catch(e){console.error(e);}
}
function buildF1Chart(metric) {
  const labels=Object.keys(f1Data.crf).filter(k=>k!=='micro_avg').map(k=>k.toUpperCase());
  const smD=labels.map(l=>{const d=f1Data.softmax[l.toLowerCase()]; return d?+(d[metric]*100).toFixed(1):0;});
  const crD=labels.map(l=>{const d=f1Data.crf[l.toLowerCase()]; return d?+(d[metric]*100).toFixed(1):0;});
  const ctx=document.getElementById('f1Chart').getContext('2d');
  if(f1Chart)f1Chart.destroy();
  f1Chart=new Chart(ctx,{type:'bar',data:{labels,datasets:[
    {label:'BiLSTM+Softmax',data:smD,backgroundColor:'rgba(241,148,63,0.5)',borderColor:'#f1943f',borderWidth:1,borderRadius:4},
    {label:'BiLSTM+CRF',data:crD,backgroundColor:'rgba(124,143,248,0.5)',borderColor:'#7c8ff8',borderWidth:1,borderRadius:4}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#9499b8',font:{size:12,family:'Inter'}}}},
    scales:{x:{ticks:{color:'#9499b8'},grid:{color:'rgba(255,255,255,0.03)'}},y:{min:0,max:100,ticks:{color:'#9499b8',callback:v=>v+'%'},grid:{color:'rgba(255,255,255,0.03)'}}}}});
}
function buildF1Table() {
  const entities=Object.keys(f1Data.crf).filter(k=>k!=='micro_avg');
  let h='';
  for(const e of entities){
    const sm=f1Data.softmax[e]||{precision:0,recall:0,f1:0,support:0}, cr=f1Data.crf[e]||{precision:0,recall:0,f1:0,support:0};
    const f1key = sm['f1-score'] !== undefined ? 'f1-score' : 'f1';
    const f1keyCr = cr['f1-score'] !== undefined ? 'f1-score' : 'f1';
    h+=`<tr style="border-bottom:1px solid var(--border)">
      <td style="padding:8px 12px;font-weight:600">${e.toUpperCase()}</td>
      <td style="padding:8px;text-align:center;color:#f1943f">${((sm.precision||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:#f1943f">${((sm.recall||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:#f1943f;font-weight:700">${((sm[f1key]||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--accent)">${((cr.precision||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--accent)">${((cr.recall||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--accent);font-weight:700">${((cr[f1keyCr]||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--text3)">${(sm.support||cr.support||0).toLocaleString()}</td></tr>`;
  }
  if(f1Data.softmax.micro_avg&&f1Data.crf.micro_avg){
    const s=f1Data.softmax.micro_avg, c=f1Data.crf.micro_avg;
    const sf=s['f1-score']!==undefined?'f1-score':'f1', cf=c['f1-score']!==undefined?'f1-score':'f1';
    h+=`<tr style="background:var(--bg3);font-weight:700"><td style="padding:8px 12px">MICRO AVG</td>
      <td style="padding:8px;text-align:center;color:#f1943f">${((s.precision||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:#f1943f">${((s.recall||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:#f1943f">${((s[sf]||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--accent)">${((c.precision||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--accent)">${((c.recall||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--accent)">${((c[cf]||0)*100).toFixed(0)}%</td>
      <td style="padding:8px;text-align:center;color:var(--text3)">${(s.support||0).toLocaleString()}</td></tr>`;
  }
  document.getElementById('f1-tbody').innerHTML = h;
}

document.querySelectorAll('.chart-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.chart-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active'); currentMetric=btn.dataset.metric;
    if(f1Data)buildF1Chart(currentMetric);
  });
});

// ── Training History ─────────────────────────────────
let histChart1=null, histChart2=null, histLoaded=false;
async function loadTrainingHistory() {
  if(histLoaded) return;
  try {
    const r=await fetch(API+'/api/training-history'); const data=await r.json();
    const smE=data.bilstm_softmax?.epochs||[], crE=data.bilstm_crf?.epochs||[];
    // Loss chart
    const ctx1=document.getElementById('lossChart');
    if(ctx1){
      if(histChart1)histChart1.destroy();
      histChart1=new Chart(ctx1,{type:'line',data:{labels:crE.map(e=>e.epoch),datasets:[
        {label:'Softmax Loss',data:smE.map(e=>e.avg_loss),borderColor:'#f1943f',backgroundColor:'rgba(241,148,63,0.1)',fill:true,tension:0.3,borderWidth:2,pointRadius:4},
        {label:'CRF Loss',data:crE.map(e=>e.avg_loss),borderColor:'#7c8ff8',backgroundColor:'rgba(124,143,248,0.1)',fill:true,tension:0.3,borderWidth:2,pointRadius:4}
      ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#9499b8',font:{family:'Inter'}}}},
        scales:{x:{title:{display:true,text:'Epoch',color:'#9499b8'},ticks:{color:'#9499b8'},grid:{color:'rgba(255,255,255,0.03)'}},
          y:{title:{display:true,text:'Loss',color:'#9499b8'},ticks:{color:'#9499b8'},grid:{color:'rgba(255,255,255,0.03)'}}}}});
    }
    // F1 chart
    const ctx2=document.getElementById('f1HistChart');
    if(ctx2){
      if(histChart2)histChart2.destroy();
      histChart2=new Chart(ctx2,{type:'line',data:{labels:crE.map(e=>e.epoch),datasets:[
        {label:'Softmax Dev F1',data:smE.map(e=>e.dev_f1),borderColor:'#f1943f',borderDash:[5,5],tension:0.3,borderWidth:2,pointRadius:4},
        {label:'Softmax Test F1',data:smE.map(e=>e.test_f1),borderColor:'#e67e22',tension:0.3,borderWidth:2,pointRadius:4},
        {label:'CRF Dev F1',data:crE.map(e=>e.dev_f1),borderColor:'#7c8ff8',borderDash:[5,5],tension:0.3,borderWidth:2,pointRadius:4},
        {label:'CRF Test F1',data:crE.map(e=>e.test_f1),borderColor:'#5c6fe8',tension:0.3,borderWidth:2,pointRadius:4}
      ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#9499b8',font:{family:'Inter'}}}},
        scales:{x:{title:{display:true,text:'Epoch',color:'#9499b8'},ticks:{color:'#9499b8'},grid:{color:'rgba(255,255,255,0.03)'}},
          y:{title:{display:true,text:'F1 (%)',color:'#9499b8'},ticks:{color:'#9499b8'},grid:{color:'rgba(255,255,255,0.03)'}}}}});
    }
    histLoaded=true;
  } catch(e){console.error(e);}
}

// ── Ablation ─────────────────────────────────────────
let ablationLoaded=false, ablChart=null;
async function loadAblation() {
  if(ablationLoaded) return;
  try {
    const r=await fetch(API+'/api/ablation-results'); const data=await r.json();
    const v=data.variants||[];
    // Table
    let h='';
    v.forEach((item,i)=>{
      const best=v.reduce((a,b)=>(b.best_dev_f1||0)>(a.best_dev_f1||0)?b:a,v[0]);
      const isBest=item===best;
      h+=`<tr${isBest?' class="ablation-highlight"':''}>
        <td style="text-align:left;padding:10px 14px;font-weight:${isBest?700:400}">${item.label}${isBest?' 🏆':''}</td>
        <td>${item.crf?'✓':'✗'}</td><td>${item.char_dim>0?'✓':'✗'}</td>
        <td style="font-weight:600;color:${isBest?'#2ecc71':'var(--text2)'}">${item.best_dev_f1?.toFixed(1)||'N/A'}%</td>
        <td style="font-weight:600;color:${isBest?'#2ecc71':'var(--text2)'}">${item.final_test_f1?.toFixed(1)||'N/A'}%</td></tr>`;
    });
    const tb=document.getElementById('ablation-tbody');
    if(tb) tb.innerHTML=h;
    // Chart
    const ctx=document.getElementById('ablationChart');
    if(ctx&&v.length){
      if(ablChart)ablChart.destroy();
      ablChart=new Chart(ctx,{type:'bar',data:{labels:v.map(x=>x.label.replace('BiLSTM+','')),datasets:[
        {label:'Dev F1',data:v.map(x=>x.best_dev_f1||0),backgroundColor:'rgba(124,143,248,0.5)',borderColor:'#7c8ff8',borderWidth:1,borderRadius:4},
        {label:'Test F1',data:v.map(x=>x.final_test_f1||0),backgroundColor:'rgba(46,204,113,0.5)',borderColor:'#2ecc71',borderWidth:1,borderRadius:4}
      ]},options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',plugins:{legend:{labels:{color:'#9499b8',font:{family:'Inter'}}}},
        scales:{x:{min:0,max:100,ticks:{color:'#9499b8',callback:v=>v+'%'},grid:{color:'rgba(255,255,255,0.03)'}},y:{ticks:{color:'#9499b8',font:{size:11}},grid:{display:false}}}}});
    }
    ablationLoaded=true;
  } catch(e){console.error(e);}
}

// ── Boundary animation ───────────────────────────────
function animateBars() { setTimeout(()=>{const b=document.getElementById('softmax-bar');if(b)b.style.width='100%';},300); }

// ── Buttons ──────────────────────────────────────────
document.getElementById('btn-analyze').addEventListener('click', doAnalyze);
document.getElementById('btn-clear').addEventListener('click', () => {
  document.getElementById('input-text').value = '';
  document.getElementById('crf-output').innerHTML = '<div class="placeholder">Nhập văn bản và bấm Phân tích</div>';
  document.getElementById('softmax-output').innerHTML = '<div class="placeholder">Nhập văn bản và bấm Phân tích</div>';
  document.getElementById('legend-box').style.display = 'none';
  document.getElementById('comparison-row').style.display = 'none';
});
document.getElementById('input-text').addEventListener('keydown', e => { if(e.key==='Enter'&&e.ctrlKey)doAnalyze(); });

// ── Init ─────────────────────────────────────────────
initParticles(); checkStatus(); loadDemoSentences();
