import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (
    StaleElementReferenceException,
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
)
import threading
from queue import Queue

BASE = "https://www.justice.gov/epstein/doj-disclosures/data-set-{}-files?page={}"
DOWNLOAD_DIR = Path.home() / "epstein files"
DOWNLOAD_DIR.mkdir(exist_ok=True)

DELAY_BETWEEN_CLICKS = 0.2  
DELAY_BETWEEN_PAGES = 0.5   
DELAY_AFTER_SCROLL = 0.1    
DELAY_AFTER_PAGE_LOAD = 1.5 
NUM_PARALLEL_TABS = 3       

options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")  
options.add_experimental_option("prefs", {
    "download.default_directory": str(DOWNLOAD_DIR),
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True,
})

downloaded = set()
download_lock = threading.Lock()


def load_existing_files():
    
    if DOWNLOAD_DIR.exists():
        for file in DOWNLOAD_DIR.glob("*.pdf"):
            downloaded.add(file.name)
        print(f"Found {len(downloaded)} existing PDFs to skip")

def create_driver():
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

def check_and_handle_age_verification(driver):
    
    try:
        
        age_verify_selectors = [
            "button:contains('Yes')",
            "button:contains('I am 18')",
            "button:contains('Confirm')",
            "a:contains('Yes')",
            ".age-verify-button",
            "#age-verify-yes",
            "[data-age-verify='yes']",
        ]
        
        
        for selector in age_verify_selectors:
            try:
                
                if "contains" in selector:
                    text = selector.split("'")[1]
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    buttons.extend(driver.find_elements(By.TAG_NAME, "a"))
                    
                    for btn in buttons:
                        try:
                            if text.lower() in btn.text.lower():
                                btn.click()
                                time.sleep(0.3)
                                return True
                        except:
                            continue
                else:
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    if element.is_displayed():
                        element.click()
                        time.sleep(0.3)
                        return True
            except:
                continue
        
        
        try:
            
            modal = driver.find_element(By.CSS_SELECTOR, ".modal, .overlay, [role='dialog']")
            if modal.is_displayed():
                
                buttons = modal.find_elements(By.TAG_NAME, "button")
                buttons.extend(modal.find_elements(By.TAG_NAME, "a"))
                
                for btn in buttons:
                    text = btn.text.lower()
                    if any(word in text for word in ["yes", "confirm", "agree", "accept", "18"]):
                        btn.click()
                        time.sleep(0.3)
                        return True
        except:
            pass
            
        return False
        
    except Exception as e:
        return False

def wait_for_pdfs(driver, timeout=10):
    
    try:
        wait = WebDriverWait(driver, timeout)
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "a[href*='.pdf'], a[href*='.PDF']")
            )
        )
        return True
    except TimeoutException:
        return False

def get_pdf_links(driver):
   
    return driver.find_elements(
        By.CSS_SELECTOR,
        "a[href*='.pdf'], a[href*='.PDF']"
    )

def scroll_to_element(driver, element):
    
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        time.sleep(DELAY_AFTER_SCROLL)
        return True
    except Exception as e:
        return False

def trigger_downloads(driver, tab_id=""):
    
    links = get_pdf_links(driver)
    print(f"[Tab {tab_id}] PDFs on page: {len(links)}")
    
    downloaded_this_page = 0
    skipped_this_page = 0
    
    for i, link in enumerate(links, start=1):
        try:
            href = link.get_attribute("href")
        except StaleElementReferenceException:
            continue
        
        if not href:
            continue
        
        name = href.split("/")[-1]
        
        
        with download_lock:
            if name in downloaded:
                skipped_this_page += 1
                continue
        
        
        check_and_handle_age_verification(driver)
        
        try:
            
            if not scroll_to_element(driver, link):
                continue
            
           
            link.click()
            
            with download_lock:
                downloaded.add(name)
            
            downloaded_this_page += 1
            time.sleep(DELAY_BETWEEN_CLICKS)
            
        except (StaleElementReferenceException, ElementClickInterceptedException) as e:
            
            if check_and_handle_age_verification(driver):
                time.sleep(0.2)
            
            
            try:
                links = get_pdf_links(driver)
                for retry_link in links:
                    retry_href = retry_link.get_attribute("href")
                    if retry_href and retry_href.split("/")[-1] == name:
                        scroll_to_element(driver, retry_link)
                        retry_link.click()
                        
                        with download_lock:
                            downloaded.add(name)
                        
                        downloaded_this_page += 1
                        time.sleep(DELAY_BETWEEN_CLICKS)
                        break
            except Exception:
                continue
    
    print(f"[Tab {tab_id}] Downloaded: {downloaded_this_page} | Skipped: {skipped_this_page}")
    
    return downloaded_this_page

def page_has_same_pdfs_as_previous(current_pdfs, previous_pdfs):
    
    if not previous_pdfs:
        return False
    
    current_set = set(current_pdfs)
    previous_set = set(previous_pdfs)
    
    return current_set == previous_set

def scrape_dataset_single_thread(driver, dataset, tab_id="", first_load=False):
    
    page = 0
    total_downloads = 0
    previous_page_pdfs = []
    consecutive_duplicate_pages = 0
    MAX_DUPLICATE_PAGES = 2
    
    while True:
        url = BASE.format(dataset, page)
        print(f"\n[Tab {tab_id}] {'='*60}")
        print(f"[Tab {tab_id}] Dataset {dataset} - Page {page}")
        print(f"[Tab {tab_id}] {'='*60}")
        
        driver.get(url)
        time.sleep(DELAY_AFTER_PAGE_LOAD)
        
        
        if first_load and page == 0:
            print(f"\n[Tab {tab_id}] This tab needs manual CAPTCHA/age verification")
            input(f"[Tab {tab_id}] Solve CAPTCHA + age verify for this tab, then press ENTER...\n")
            first_load = False
        
        
        check_and_handle_age_verification(driver)
        time.sleep(0.3)
        
        
        if not wait_for_pdfs(driver):
            print(f"[Tab {tab_id}] No PDFs found on page {page}, ending dataset {dataset}")
            break
        
        
        links = get_pdf_links(driver)
        current_page_pdfs = []
        for link in links:
            try:
                href = link.get_attribute("href")
                if href:
                    current_page_pdfs.append(href.split("/")[-1])
            except:
                continue
        
        
        if page_has_same_pdfs_as_previous(current_page_pdfs, previous_page_pdfs):
            consecutive_duplicate_pages += 1
            
            if consecutive_duplicate_pages >= MAX_DUPLICATE_PAGES:
                print(f"[Tab {tab_id}] Hit {MAX_DUPLICATE_PAGES} consecutive duplicate pages")
                print(f"[Tab {tab_id}] Ending dataset {dataset} at page {page - 1}")
                break
        else:
            consecutive_duplicate_pages = 0
        
        previous_page_pdfs = current_page_pdfs
        
        
        page_downloads = trigger_downloads(driver, tab_id)
        total_downloads += page_downloads
        
        
        page += 1
        time.sleep(DELAY_BETWEEN_PAGES)
    
    print(f"\n[Tab {tab_id}] {'='*60}")
    print(f"[Tab {tab_id}] Dataset {dataset} complete: {total_downloads} new files downloaded")
    print(f"[Tab {tab_id}] {'='*60}\n")
    return total_downloads

def worker_thread(task_queue, results_queue, tab_id):
    
    driver = create_driver()
    first_dataset = True
    
    try:
        while True:
            try:
                dataset = task_queue.get(timeout=1)
                if dataset is None:  
                    break
                
                downloads = scrape_dataset_single_thread(driver, dataset, str(tab_id), first_load=first_dataset)
                first_dataset = False  
                results_queue.put((dataset, downloads))
                task_queue.task_done()
                
            except Exception as e:
                print(f"[Tab {tab_id}] Error: {e}")
                import traceback
                traceback.print_exc()
                task_queue.task_done()
                
    finally:
        driver.quit()

def dataset_exists(driver, dataset):
    
    try:
        url = BASE.format(dataset, 0)
        driver.get(url)
        time.sleep(DELAY_AFTER_PAGE_LOAD)
        
        check_and_handle_age_verification(driver)
        time.sleep(0.3)
        
        if wait_for_pdfs(driver):
            return True
        else:
            return False
            
    except Exception as e:
        return False

def main_multithreaded():
    
    print("Epstein Files Scraper (Multi-Threaded)")
    print("="*60)
    print(f"Using {NUM_PARALLEL_TABS} parallel tabs")
    print(f"Click delay: {DELAY_BETWEEN_CLICKS}s")
    print(f"Page delay: {DELAY_BETWEEN_PAGES}s")
    print("="*60)
    
    
    load_existing_files()
    
    
    print("\nFinding available datasets...")
    main_driver = create_driver()
    available_datasets = []
    
    for dataset in range(1, 20):  
        if dataset_exists(main_driver, dataset):
            available_datasets.append(dataset)
            print(f"Found dataset {dataset}")
        else:
            print(f"Dataset {dataset} not found, stopping search")
            break
    
    main_driver.quit()
    
    if not available_datasets:
        print("No datasets found!")
        return
    
    print(f"\nFound {len(available_datasets)} datasets: {available_datasets}")
    print(f"\nStarting multi-threaded download with {NUM_PARALLEL_TABS} tabs...")
    print(f"Each tab will process datasets sequentially to avoid conflicts\n")
    
   
    tab_assignments = [[] for _ in range(NUM_PARALLEL_TABS)]
    for i, dataset in enumerate(available_datasets):
        tab_assignments[i % NUM_PARALLEL_TABS].append(dataset)
    
    for i, datasets in enumerate(tab_assignments):
        print(f"Tab {i+1} will process datasets: {datasets}")
    
    
    task_queues = [Queue() for _ in range(NUM_PARALLEL_TABS)]
    results_queue = Queue()
    
    
    for i, datasets in enumerate(tab_assignments):
        for dataset in datasets:
            task_queues[i].put(dataset)
    
    
    threads = []
    for i in range(NUM_PARALLEL_TABS):
        t = threading.Thread(target=worker_thread, args=(task_queues[i], results_queue, i+1))
        t.start()
        threads.append(t)
        time.sleep(3)  
    
   
    for queue in task_queues:
        queue.join()
    
    
    for queue in task_queues:
        queue.put(None)
    
    
    for t in threads:
        t.join()
    
    
    total_files = 0
    while not results_queue.empty():
        dataset, downloads = results_queue.get()
        total_files += downloads
    
    print("\n" + "="*60)
    print(f" Done! Downloaded {total_files} new files")
    print(f"Total files in directory: {len(downloaded)}")
    print(f"Files saved to: {DOWNLOAD_DIR}")
    print("="*60)

def main_single_threaded():
    
    print("Epstein Files Scraper (Single-Threaded, Optimized)")
    print("="*60)
    print(f"Click delay: {DELAY_BETWEEN_CLICKS}s")
    print(f"Page delay: {DELAY_BETWEEN_PAGES}s")
    print("="*60)
    
    driver = create_driver()
    wait = WebDriverWait(driver, 10)
    
   
    load_existing_files()
    
    dataset = 1
    total_files = 0
    
    while True:
        try:
            print(f"\n\nStarting Dataset {dataset}...")
            dataset_downloads = scrape_dataset_single_thread(driver, dataset)
            total_files += dataset_downloads
            
            
            print(f"\nChecking for Dataset {dataset + 1}...")
            time.sleep(0.5)
            
            if dataset_exists(driver, dataset + 1):
                dataset += 1
                time.sleep(1)
            else:
                print(f"No more datasets found after Dataset {dataset}")
                break
                
        except Exception as e:
            import traceback
            print(f"\n[ERROR] While processing dataset {dataset}: {e}")
            traceback.print_exc()
            
            try:
                response = input("\nTry next dataset? (y/n): ")
                if response.lower() == 'y':
                    dataset += 1
                    continue
                else:
                    break
            except:
                break
    
    print("\n" + "="*60)
    print(f"Done! Downloaded {total_files} new files")
    print(f"Total files in directory: {len(downloaded)}")
    print(f"Files saved to: {DOWNLOAD_DIR}")
    print("="*60)
    
    time.sleep(5)
    driver.quit()

if __name__ == "__main__":
   
    print("Choose scraping mode:")
    print("1. Single-threaded (safer, recommended)")
    print("2. Multi-threaded (faster, uses multiple tabs)")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            main_multithreaded()
        else:
            main_single_threaded()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except:
        
        main_single_threaded()