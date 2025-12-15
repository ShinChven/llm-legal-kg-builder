from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from src.db.db_connection import db_connection
import psycopg2

BASE_URL = "https://www.legislation.govt.nz/subscribe/act"

def get_links(driver, url):
    driver.get(url)
    links = driver.find_elements(By.CSS_SELECTOR, "ul > li.directory > a")
    return [link.get_attribute('href') for link in links]

def get_file_links(driver, url):
    driver.get(url)
    links = driver.find_elements(By.CSS_SELECTOR, "ul > li.file > a")
    return [link.get_attribute('href') for link in links]

def main():
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    conn = None
    try:
        conn = db_connection.get_connection()
        cursor = conn.cursor()

        jurisdiction_links = get_links(driver, BASE_URL)
        for jurisdiction_link in jurisdiction_links:
            year_links = get_links(driver, jurisdiction_link)
            for year_link in year_links:
                act_links = get_links(driver, year_link)
                for act_link in act_links:
                    version_links = get_links(driver, act_link)
                    for version_link in version_links:
                        file_links = get_file_links(driver, version_link)
                        xml_link = None
                        pdf_link = None
                        for file_link in file_links:
                            if file_link.endswith('.xml'):
                                xml_link = file_link
                            elif file_link.endswith('.pdf'):
                                pdf_link = file_link

                        if xml_link:
                            version = version_link.split('/')[-1]
                            cursor.execute(
                                """INSERT INTO legislations (xml_link, pdf_link, source, doc_type, version)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (xml_link) DO NOTHING;""",
                                (xml_link, pdf_link, 'legislation.govt.nz', 'act', version)
                            )
                            conn.commit()
                            print(f"Indexed: {xml_link}")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while processing: {error}")
    finally:
        if conn:
            db_connection.release_connection(conn)
        driver.quit()

if __name__ == "__main__":
    main()
