from playwright.sync_api import sync_playwright
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
import urllib.request
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium'),headless=True,args=['--no-sandbox'])
 context=b.new_context(viewport={'width':1440,'height':1000})
 def trusted_fetch(route):
  try:
   with urllib.request.urlopen(route.request.url,timeout=60) as r:
    headers={k:v for k,v in r.headers.items() if k.lower() not in ['content-length','content-encoding','transfer-encoding']}
    route.fulfill(status=r.status,headers=headers,body=r.read())
  except Exception as exc:
   print('PROXY FETCH FAILURE',type(exc).__name__)
   route.abort()
 if os.environ.get('PAGES_TEST_PROXY')=='1':
  context.route('https://cdn.jsdelivr.net/**',trusted_fetch)
  context.route('https://raw.githubusercontent.com/**',trusted_fetch)
 page=context.new_page()
 errors=[]
 page.on('pageerror',lambda e: errors.append(str(e)))
 page.on('console',lambda msg: print('BROWSER:',msg.text) if msg.type=='error' else None)
 page.goto(os.environ.get('PAGES_URL','http://127.0.0.1:8765'))
 page.locator('#analysis:not(.hidden)').wait_for()
 page.locator('#predict-form button').click()
 print('Prediction',page.locator('#prediction-value').inner_text())
 page.locator('#compare-methods').click()
 page.screenshot(path='/tmp/pages-desktop.png',full_page=True)
 page.locator('#problem').select_option('appliances')
 page.locator('#analysis-title').filter(has_text='Appliances').wait_for()
 page.locator('#eval-split').select_option('test')
 print('Appliance',page.locator('#diagnostic-note').inner_text())
 page.locator('[data-view=research]').click()
 page.locator('#original-form button').click()
 print('Original',page.locator('#original-value').inner_text(),page.locator('#original-interval').inner_text())
 page.locator('[data-view=custom]').click()
 page.locator('#csv-file').set_input_files(str(ROOT/'data/reference/50_Startups.csv'))
 page.locator('#dataset-config:not(.hidden)').wait_for()
 page.locator('#train-custom').click()
 print('Waiting for browser Python...')
 page.locator('#analysis:not(.hidden)').wait_for(timeout=180000)
 print('Custom loaded',page.locator('#status').inner_text())
 page.locator('#predict-form button').click()
 print('Custom prediction',page.locator('#prediction-value').inner_text())
 page.screenshot(path='/tmp/pages-custom.png',full_page=True)
 page.locator('#cancel-custom').click()
 page.locator('#load-url').click()
 page.locator('#dataset-config:not(.hidden)').wait_for()
 assert page.locator('#preview tbody tr').count()==6
 print('PASS: public CSV URL loading')
 assert not errors,errors
 mobile=b.new_page(viewport={'width':390,'height':844},is_mobile=True)
 mobile.goto(os.environ.get('PAGES_URL','http://127.0.0.1:8765'));mobile.locator('#analysis:not(.hidden)').wait_for()
 assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile overflow'
 mobile.screenshot(path='/tmp/pages-mobile.png',full_page=True)
 print('PASS: static problems, research, actual CSV upload, browser training, live prediction, responsive layout')
 b.close()
