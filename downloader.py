import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 1. Setup download directory
download_dir = os.path.abspath("./thermal_matrices")
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# 2. Configure Chrome Options to prevent popups and save files automatically
options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
options.add_experimental_option("prefs", prefs)

# Initialize the automated Chrome browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    # 3. Manual Login Pause
    driver.get("https://visual.ic.uff.br/dmi/")
    print("\n" + "="*60)
    print("ACTION REQUIRED:")
    print("1. Log in manually in the Chrome browser window that just opened.")
    print("2. Once you are securely logged in, come back to this terminal.")
    print("3. Press ENTER here to unleash the downloader script.")
    print("="*60 + "\n")
    input()

    # 4. Loop through the adjusted patient ranges
    start_id = 72
    end_id = 425

    for patient_id in range(start_id, end_id + 1):
        try:
            target_url = f"https://visual.ic.uff.br/dmi/prontuario/details.php?id={patient_id}"
            print(f"\nNavigating to Patient {patient_id}...")
            driver.get(target_url)
            time.sleep(2)  # Generous page-load buffer for stable execution

            # Find all text matrix links (.txt) present on the page
            matrix_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.txt')]")

            if not matrix_links:
                print(f"--> [Missing/Empty] No links found for ID {patient_id}. Skipping.")
                continue

            print(f"--> Found {len(matrix_links)} matrix files across all dates. Extracting...")
            
            for link in matrix_links:
                try:
                    link.click()
                    time.sleep(1.0)  # 1-second delay to safely clear Chrome's download queue
                except Exception as click_error:
                    print(f"    Failed to click an individual link for patient {patient_id}: {click_error}")

        except Exception as page_error:
            # Safe recovery if a network dip, timeout, or server crash happens
            print(f"XX Unexpected issue loading page for Patient {patient_id}: {page_error}")
            print("Moving to next patient record to protect script execution...")
            continue

    # Final wrap-up buffer to let trailing downloads finalize writing to disk
    print("\nWaiting for final files to complete processing...")
    time.sleep(8)

finally:
    print("\nSequence fully completed! Closing browser.")
    driver.quit()