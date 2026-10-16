import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parseCSV,predictModel,predictOriginal} from '../web/core.mjs';
const sample=readFileSync(new URL('../data/reference/50_Startups.csv',import.meta.url),'utf8');
assert.equal(parseCSV(sample).rows.length,50);
assert.throws(()=>parseCSV('a,a\n1,2'),/不重複/);
assert.throws(()=>parseCSV('a,b\n"bad'),/引號/);
assert.throws(()=>parseCSV('a,b\n1,2'),/20/);
const booleanSchema={encoded:[{kind:'category',field:'flag',value:true}]};
assert.equal(predictModel({intercept:1,cols:[0],coef:[2]},booleanSchema,{flag:true}),3);
for(const id of ['startups','appliances']){
 const data=JSON.parse(readFileSync(new URL(`../web/data/${id}.json`,import.meta.url)));
 const values=Object.fromEntries(data.schema.inputs.map(i=>[i.name,i.default]));
 for(const group of Object.values(data.methods))for(const model of group.models){
   assert.ok(Number.isFinite(predictModel(model,data.schema,values)));
   assert.equal(model.names.length,model.k);
   assert.equal(model.coef.length,model.k);
   if(!model.inference.error)assert.equal(model.inference.table.length,model.k+1);
 }
}
const research=JSON.parse(readFileSync(new URL('../web/data/research.json',import.meta.url)));
assert.equal(research.validation.length,32);
assert.equal(new Set(research.validation.map(r=>r.loss_function)).size,4);
assert.ok(Number.isFinite(predictOriginal(research.original,Object.fromEntries(research.original.names.map((n,i)=>[n,research.original.defaults[i]])))));
console.log('PASS: CSV parsing, typed categories, model payload dimensions, 32-model preservation');
