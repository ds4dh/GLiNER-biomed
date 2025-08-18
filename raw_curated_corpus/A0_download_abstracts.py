import argparse
import csv
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep


def fetch_pubmed_data(mesh_query, start_date, end_date, output_csv):
    # Base URL and settings for esearch and efetch
    base_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    db = 'db=pubmed'

    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y.%m.%d")
    end_date = datetime.strptime(end_date, "%Y.%m.%d")
    current_date = start_date

    # Prepare CSV for writing
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["PMID", "Date", "Title", "Abstract"])
        writer.writeheader()

        # Loop through each day in the date range
        while current_date <= end_date:
            date_str = current_date.strftime("%Y/%m/%d")
            query = f'({mesh_query}) AND ((fha[Filter]) AND ({date_str}:{date_str}[pdat]))'

            # Esearch settings
            search_eutil = 'esearch.fcgi?'
            search_term = '&term=' + urllib.parse.quote(query)
            search_usehistory = '&usehistory=y'
            search_rettype = '&rettype=json'
            search_url = base_url + search_eutil + db + search_term + search_usehistory + search_rettype

            try:
                f = urllib.request.urlopen(search_url)
                search_data = f.read().decode('utf-8')
            except Exception as e:
                print(f"Error during Esearch for {date_str}: {e}")
                current_date += timedelta(days=1)
                continue

            # Extract total abstract count, QueryKey, and WebEnv
            try:
                total_abstract_count = int(re.findall(r"<Count>(\d+?)</Count>", search_data)[0])
                if total_abstract_count == 0:
                    print(f"No abstracts found for {date_str}. Skipping...")
                    current_date += timedelta(days=1)
                    continue

                fetch_webenv = "&WebEnv=" + re.findall(r"<WebEnv>(\S+)</WebEnv>", search_data)[0]
                fetch_querykey = "&query_key=" + re.findall(r"<QueryKey>(\d+?)</QueryKey>", search_data)[0]
            except Exception as e:
                print(f"Error parsing Esearch results for {date_str}: {e}")
                current_date += timedelta(days=1)
                continue

            print(f"Total abstracts to fetch for {date_str}: {total_abstract_count}")

            # Efetch settings
            fetch_eutil = 'efetch.fcgi?'
            retmax = 9999  # Maximum abstracts per request
            retstart = 0
            fetch_retmode = "&retmode=xml"
            fetch_rettype = "&rettype=abstract"

            # Loop to fetch all abstracts
            all_data = []  # To store abstracts, titles, and dates
            while retstart < min(total_abstract_count, 9999):  # Limit to 9999 abstracts per day
                fetch_retstart = "&retstart=" + str(retstart)
                fetch_retmax = "&retmax=" + str(retmax)
                fetch_url = (base_url + fetch_eutil + db + fetch_querykey + fetch_webenv +
                             fetch_retstart + fetch_retmax + fetch_retmode + fetch_rettype)
                print(f"Fetching abstracts {retstart + 1} to {min(retstart + retmax, total_abstract_count)} for {date_str}...")

                retries = 0
                max_retries = 5
                while retries < max_retries:
                    try:
                        f = urllib.request.urlopen(fetch_url)
                        fetch_data = f.read().decode('utf-8')

                        # Parse the XML data
                        root = ET.fromstring(fetch_data)

                        # Extract abstract data
                        for article in root.findall(".//PubmedArticle"):
                            pmid = None
                            title = None
                            abstract_text = None
                            pub_date = date_str  # Use the known date

                            # Extract PMID
                            pmid_element = article.find('.//MedlineCitation/PMID')
                            if pmid_element is not None and pmid_element.text:
                                pmid = pmid_element.text.strip()
                            else:
                                print("Missing PMID for an article. Skipping...")
                                continue

                            # Extract title
                            article_title = article.find(".//ArticleTitle")
                            if article_title is not None and article_title.text:
                                title = article_title.text.strip()
                            else:
                                print("Missing title for an article. Skipping...")
                                continue

                            # Extract abstract text
                            abstract = article.find(".//Abstract")
                            if abstract is not None:
                                # Case 1: Unstructured abstract
                                if len(abstract.findall(".//AbstractText")) == 1:
                                    abstract_section = abstract.find(".//AbstractText")
                                    if abstract_section is not None:
                                        abstract_text = "".join(abstract_section.itertext()).strip()
                                # Case 2: Structured abstract
                                else:
                                    abstract_text_parts = []
                                    for abstract_section in abstract.findall(".//AbstractText"):
                                        label = abstract_section.attrib.get("Label")
                                        text = "".join(abstract_section.itertext()).strip()
                                        if text and label:
                                            abstract_text_parts.append(f"{label.strip().upper()}: {text}")
                                        elif text:
                                            abstract_text_parts.append(text)
                                    abstract_text = "\n".join(abstract_text_parts) if abstract_text_parts else None

                            if not abstract_text:
                                continue

                            # Append data to the list
                            all_data.append({
                                "PMID": pmid,
                                "Date": pub_date.strip(),
                                "Title": title,
                                "Abstract": abstract_text.strip()
                            })

                        break  # Exit retry loop after successful fetch

                    except urllib.error.HTTPError as e:
                        retries += 1
                        print(f"HTTP Error {e.code}: {e.reason}. Retrying ({retries}/{max_retries})...")
                        sleep(2 ** retries)  # Exponential backoff
                        if retries == max_retries:
                            print(f"Max retries reached for batch starting at {retstart}. Skipping batch...")
                            break
                    except Exception as e:
                        retries += 1
                        print(f"Unexpected error: {e}. Retrying ({retries}/{max_retries})...")
                        sleep(2 ** retries)
                        if retries == max_retries:
                            print(f"Max retries reached for batch starting at {retstart}. Skipping batch...")
                            break

                retstart += retmax

            # Write data for the day
            writer.writerows(all_data)
            print(f"Wrote {len(all_data)} records for {date_str} to {output_csv}.")

            # Move to the next day
            current_date += timedelta(days=1)

    print(f"Data collection complete. Results saved to {output_csv}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts and save them to a CSV file.")
    parser.add_argument("--mesh_query", type=str, default='"pathological conditions, signs and symptoms"[MeSH Terms]',
                        help="PubMed MeSH query string.")
    parser.add_argument("--start_date", type=str, default="2024.01.01",
                        help="Start date for the query in YYYY.MM.DD format.")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y.%m.%d"),
                        help="End date for the query in YYYY.MM.DD format.")
    parser.add_argument("--output_csv", type=str, default="./data/pubmed_raw_abstracts.csv",
                        help="Name of the output CSV file.")

    args = parser.parse_args()
    fetch_pubmed_data(args.mesh_query, args.start_date, args.end_date, args.output_csv)