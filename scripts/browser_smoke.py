from playwright.sync_api import sync_playwright
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium'),headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.goto(os.environ.get('EXPLORER_URL','http://127.0.0.1:8502'))
    page.get_by_role('button',name='計算／更新 K 分析',exact=True).click()
    page.get_by_text('OLS 方程式與統計推論',exact=True).wait_for()
    page.get_by_role('button',name='計算預測',exact=True).click()
    page.get_by_text('預測值',exact=True).first.wait_for()
    page.screenshot(path='/tmp/explorer-startups.png',full_page=True)
    page.get_by_role('link',name='自訂資料工作區').click()
    page.locator('input[type=file]').set_input_files(str(ROOT/'data/reference/50_Startups.csv'))
    page.get_by_role('button',name='載入上傳資料',exact=True).click()
    page.get_by_role('button',name='計算／更新 K 分析',exact=True).click()
    page.get_by_text('OLS 方程式與統計推論',exact=True).wait_for()
    assert page.get_by_test_id('stException').count()==0
    page.screenshot(path='/tmp/explorer-custom.png',full_page=True)
    print('PASS: actual browser navigation, CSV upload, model training, live prediction, no Streamlit exceptions')
    browser.close()
