import os
import requests
import psycopg2
from src.db.db_connection import db_connection
from bs4 import BeautifulSoup
from io import BytesIO
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
MAX_WORKERS = 3  # Number of concurrent threads for scraping


def parse_date(date_str):
    """Convert date string to None if it's empty, 'nulldate', or invalid"""
    if not date_str or date_str.strip() == '' or date_str.lower() == 'nulldate':
        return None
    return date_str


def download_file(url, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, url.split('/')[-1])
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return f.read()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return response.content


def process_record(record, lock):
    """Process a single record with thread-safe database operations"""
    record_id, xml_link, pdf_link = record
    conn = None

    try:
        # Get a separate connection for this thread
        conn = db_connection.get_connection()
        cursor = conn.cursor()

        # Download and process XML
        if xml_link:
            xml_content = download_file(xml_link, f'./subscribe/act/{'/'.join(xml_link.split('/')[4:-1])}')
            soup = BeautifulSoup(xml_content, 'xml')

            title = None
            act_id = None
            in_amend = None
            ird_numbering = None
            year = None
            act_no = None
            act_type = None
            date_as_at = None
            date_assent = None
            date_terminated = None
            date_first_valid = None
            stage = None
            terminated = None
            dlm = None

            act_tag = soup.find('act')
            if act_tag:
                cover_tag = act_tag.find('cover')
                if cover_tag:
                    title_tag = cover_tag.find('title')
                    if title_tag:
                        title = title_tag.text

                act_id = act_tag.get('id')
                in_amend = act_tag.get('in.amend') == 'true'
                ird_numbering = act_tag.get('irdnumbering')
                year_str = act_tag.get('year')
                if year_str and year_str.isdigit():
                    year = int(year_str)
                act_no = act_tag.get('act.no')
                act_type = act_tag.get('act.type')
                date_as_at = parse_date(act_tag.get('date.as.at'))
                date_assent = parse_date(act_tag.get('date.assent'))
                date_terminated = parse_date(act_tag.get('date.terminated'))
                date_first_valid = parse_date(act_tag.get('date.first.valid'))
                stage = act_tag.get('stage')
                terminated = act_tag.get('terminated')

                # dlm should be the id from the cover tag itself
                if cover_tag:
                    dlm = cover_tag.get('id')

            # Extract text from XML by removing all tags
            text = soup.get_text(separator=' ', strip=True)
            word_count = len(text.split()) if text else 0

            cursor.execute(
                """UPDATE legislations SET
                   title = %s, act_id = %s, in_amend = %s, ird_numbering = %s, year = %s, act_no = %s, act_type = %s,
                   date_as_at = %s, date_assent = %s, date_terminated = %s, date_first_valid = %s, stage = %s,
                   terminated = %s, dlm = %s, xml = %s, text = %s, word_count = %s
                   WHERE id = %s;""",
                (title, act_id, in_amend, ird_numbering, year, act_no, act_type, date_as_at, date_assent,
                 date_terminated, date_first_valid, stage, terminated, dlm, xml_content.decode('utf-8'), text, word_count, record_id)
            )

        # Download PDF file (but don't extract text from it)
        if pdf_link:
            pdf_content = download_file(pdf_link, f'./subscribe/act/{'/'.join(pdf_link.split('/')[4:-1])}')

        conn.commit()

        # Thread-safe printing
        with lock:
            print(f"Thread {threading.current_thread().name}: Scraped and updated record {record_id}")

        return f"Success: {record_id}"

    except Exception as e:
        if conn:
            conn.rollback()
        with lock:
            print(f"Thread {threading.current_thread().name}: Error processing record {record_id}: {e}")
        return f"Error: {record_id} - {str(e)}"

    finally:
        if conn:
            db_connection.release_connection(conn)

def main():
    conn = None
    try:
        # Get initial connection to fetch records
        conn = db_connection.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, xml_link, pdf_link FROM legislations WHERE xml IS NULL AND text IS NULL")
        records = cursor.fetchall()

        # Release the initial connection as threads will get their own
        db_connection.release_connection(conn)
        conn = None

        if not records:
            print("No records to process")
            return

        print(f"Found {len(records)} records to process")

        # Create a lock for thread-safe printing
        print_lock = threading.Lock()

        # Use ThreadPoolExecutor with configurable number of threads
        start_time = time.time()
        successful_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Scraper") as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(process_record, record, print_lock): record
                for record in records
            }

            # Process completed tasks
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    result = future.result()
                    if result.startswith("Success"):
                        successful_count += 1
                    else:
                        error_count += 1
                except Exception as exc:
                    with print_lock:
                        print(f"Record {record[0]} generated an exception: {exc}")
                    error_count += 1

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n=== Scraping Complete ===")
        print(f"Total records processed: {len(records)}")
        print(f"Successful: {successful_count}")
        print(f"Errors: {error_count}")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Average time per record: {duration/len(records):.2f} seconds")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while connecting to PostgreSQL: {error}")
    finally:
        if conn:
            db_connection.release_connection(conn)

if __name__ == "__main__":
    main()
