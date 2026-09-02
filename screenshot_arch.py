from playwright.sync_api import sync_playwright
import os

def capture_screenshot():
    html_path = f"file:///{os.path.abspath('architecture_ui.html').replace(chr(92), '/')}"
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1450, "height": 950})
        
        # Load the local HTML file
        page.goto(html_path)
        
        # Wait a bit for the leader-lines and animations to settle
        page.wait_for_timeout(1500)
        
        # Take the screenshot
        output_path = "sih_architecture_extraordinary.png"
        page.screenshot(path=output_path, full_page=True)
        
        print(f"Screenshot saved to {output_path}")
        browser.close()

if __name__ == "__main__":
    capture_screenshot()
