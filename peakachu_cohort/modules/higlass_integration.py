import logging
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
# from selenium.webdriver.firefox.service import Service as FirefoxService
# from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.firefox import GeckoDriverManager

logger = logging.getLogger(__name__)

def capture_higlass_screenshot(
    higlass_url: str,
    output_path: str,
    viewport_size: tuple[int, int] = (1920, 1080),
    timeout: int = 60,
    wait_selector: str = ".higlass-view-container .track-renderer-div", # Example selector
    loading_selector: str = ".loading-indicator", # Example selector
    final_wait_seconds: int = 2
):
    """
    Captures a screenshot of a HiGlass visualization using Selenium.

    Args:
        higlass_url: URL to the HiGlass view.
        output_path: Path to save the screenshot (e.g., 'screenshot.png').
        viewport_size: Tuple of (width, height) for the browser viewport.
        timeout: Maximum wait time in seconds for HiGlass elements to appear/disappear.
        wait_selector: CSS selector for a key element indicating HiGlass view is present.
                       Needs validation against actual HiGlass DOM.
        loading_selector: CSS selector for a loading indicator to wait for its disappearance.
                          Needs validation against actual HiGlass DOM.
        final_wait_seconds: Small additional delay after waits to ensure rendering completes.

    Returns:
        Path to the saved screenshot if successful, None otherwise.

    Raises:
        TimeoutException: If waiting for elements exceeds the timeout.
        Exception: For other WebDriver or screenshot errors.
    """
    logger.info(f"Attempting to capture screenshot of {higlass_url} to {output_path}")

    # Configure Chrome options for headless operation
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--window-size={viewport_size[0]},{viewport_size[1]}")
    chrome_options.add_argument("--disable-gpu")  # Often needed for headless
    chrome_options.add_argument("--no-sandbox") # Often needed in containerized environments
    chrome_options.add_argument("--disable-dev-shm-usage") # Overcomes limited resource problems

    driver = None
    try:
        logger.debug("Initializing Chrome WebDriver...")
        # Initialize the WebDriver using webdriver-manager
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options
        )
        logger.debug(f"WebDriver initialized. Navigating to {higlass_url}...")

        # Navigate to the HiGlass view
        driver.get(higlass_url)
        logger.debug("Navigation complete. Waiting for HiGlass elements...")

        # --- Wait for HiGlass to Load ---
        # 1. Wait for a core HiGlass container element to be visible
        logger.debug(f"Waiting up to {timeout}s for element '{wait_selector}' to be visible.")
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
        )
        logger.debug(f"Element '{wait_selector}' found.")

        # 2. Wait for any loading indicators to disappear (if applicable)
        if loading_selector:
            try:
                logger.debug(f"Waiting up to {timeout}s for element '{loading_selector}' to become invisible.")
                WebDriverWait(driver, timeout).until(
                    EC.invisibility_of_element_located((By.CSS_SELECTOR, loading_selector))
                )
                logger.debug(f"Element '{loading_selector}' is invisible.")
            except Exception:
                # It's possible the loading indicator never appeared or disappeared quickly
                logger.warning(f"Loading indicator '{loading_selector}' did not become invisible or wasn't found within timeout. Proceeding anyway.")
        else:
             logger.debug("No loading selector specified, skipping wait for invisibility.")

        # 3. Optional final wait for rendering/animations
        if final_wait_seconds > 0:
            logger.debug(f"Pausing for an additional {final_wait_seconds}s for final rendering...")
            time.sleep(final_wait_seconds)

        # --- Take the Screenshot ---
        logger.info("Taking screenshot...")
        success = driver.save_screenshot(output_path)
        if success:
            logger.info(f"Screenshot successfully saved to {output_path}")
            return output_path
        else:
            logger.error(f"Failed to save screenshot to {output_path}")
            return None

    except Exception as e:
        logger.error(f"An error occurred during screenshot capture: {e}", exc_info=True)
        # Optionally re-raise or handle specific exceptions
        raise
    finally:
        # --- Cleanup ---
        if driver:
            logger.debug("Quitting WebDriver.")
            driver.quit()

# Example Usage (for testing purposes, can be removed later)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Replace with a real HiGlass URL for testing
    test_url = "https://higlass.io/app/?config=CQFFX-aPTaenR3DaNf8wDA"
    test_output = "higlass_test_screenshot.png"

    try:
        result_path = capture_higlass_screenshot(test_url, test_output)
        if result_path:
            print(f"Screenshot saved to: {result_path}")
        else:
            print("Screenshot capture failed.")
    except Exception as e:
        print(f"An error occurred: {e}") 