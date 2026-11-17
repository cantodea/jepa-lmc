const models = [[0,0],[0,1],[1,0],[1,1]];
let cases = 0, weights = 0, paired = 0;
for (const p0 of [-1,0,1]) for (const p1 of [-1,0,1]) {
 const p=[p0,p1], hs=models.filter(f=>p.every((v,s)=>v<0||f[s]===v));
 for(let len=1;len<=4;len++) for(let bits=0;bits<2**(len+1);bits++) {
  const seq=Array.from({length:len+1},(_,i)=>(bits>>i)&1); const req=new Map(); let conflict=false;
  for(let i=0;i<len;i++){if(req.has(seq[i])&&req.get(seq[i])!==seq[i+1]) conflict=true;req.set(seq[i],seq[i+1]);}
  const red=conflict||Array.from(req).some(([s,t])=>p[s]>=0&&p[s]!==t);
  const accepting=hs.filter(f=>seq.slice(0,-1).every((s,i)=>f[s]===seq[i+1])).length;
  if(red!==(accepting===0))throw Error('criterion');
  if(!red){let m=p.filter(x=>x<0).length,k=Array.from(req.keys()).filter(s=>p[s]<0).length;if(accepting!==2**(m-k))throw Error('count');}
  cases++;
 }
}
for(let len=1;len<=4;len++) for(let bits=0;bits<2**(len+1);bits++) {
 const seq=Array.from({length:len+1},(_,i)=>(bits>>i)&1), req=new Map();let conflict=false;
 for(let i=0;i<len;i++){if(req.has(seq[i])&&req.get(seq[i])!==seq[i+1])conflict=true;req.set(seq[i],seq[i+1]);}
 const q=[[.2,.8],[.6,.4]];
 const actual=models.filter(f=>seq.slice(0,-1).every((s,i)=>f[s]===seq[i+1])).reduce((t,f)=>t+q[0][f[0]]*q[1][f[1]],0);
 const expected=conflict?0:Array.from(req).reduce((t,[s,u])=>t*q[s][u],1);
 if(Math.abs(actual-expected)>1e-12)throw Error('probability'); weights++;
}
for(const f of models)for(let start=0;start<2;start++)for(let len=1;len<=4;len++){
 const g=[start];for(let i=0;i<len;i++)g.push(f[g.at(-1)]);
 const hs=models.filter(h=>g.slice(0,-1).every((s,i)=>h[s]===g[i+1]));
 for(let j=1;j<=len;j++){const b=[...g];b[j]=1-b[j];if(hs.some(h=>b.slice(0,-1).every((s,i)=>h[s]===b[i+1])))throw Error('paired');paired++;}
}
console.log(JSON.stringify({criterion_and_count_cases:cases,semantic_probability_cases:weights,paired_redundancy_cases:paired,failures:0,scope:'exhaustive two-state one-action traces lengths 1..4; all nine partial good maps; four total transition functions'},null,2));
