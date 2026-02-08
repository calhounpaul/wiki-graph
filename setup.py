#!/usr/bin/env python3
"""
Setup script for MediaWiki dataset.
Downloads the XML dump and extracts pages if not already present.

Configure WIKI_NAME, XML_URL, and XML_FILENAME below for your wiki.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import re

# =============================================================================
# CONFIGURE THESE FOR YOUR WIKI
# =============================================================================
WIKI_NAME = "Memory Alpha"  # Display name for your wiki

# XML dump URL - find yours at:
# - Fandom: https://s3.amazonaws.com/wikia_xml_dumps/{a}/{ab}/{wikiname}_pages_current.xml.7z
# - Wikipedia: https://dumps.wikimedia.org/
XML_URL = "https://s3.amazonaws.com/wikia_xml_dumps/e/en/enmemoryalpha_pages_current.xml.7z"
XML_FILENAME = "enmemoryalpha_pages_current.xml"  # Filename inside the archive
# =============================================================================

# Paths (auto-configured from BASE_DIR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
PAGES_DIR = os.path.join(DATA_DIR, "pages")

XML_7Z = os.path.join(EXTRACTED_DIR, os.path.basename(XML_URL))
XML_FILE = os.path.join(EXTRACTED_DIR, XML_FILENAME)

# MediaWiki XML namespace (0.10 and 0.11 are common)
NS = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}


def check_dependencies():
    """Check if required tools are installed."""
    try:
        subprocess.run(["7z", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 7zip not found. Install with: sudo apt install p7zip-full")
        sys.exit(1)


def download_dump():
    """Download the XML dump if not present."""
    if os.path.exists(XML_FILE):
        print(f"XML file already exists: {XML_FILE}")
        return True

    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    if not os.path.exists(XML_7Z):
        print(f"Downloading {WIKI_NAME} dump...")
        print(f"  URL: {XML_URL}")
        result = subprocess.run(
            ["wget", "-O", XML_7Z, XML_URL],
            capture_output=False
        )
        if result.returncode != 0:
            print("Error: Download failed")
            return False
        print("Download complete.")

    # Extract
    print("Extracting 7z archive...")
    result = subprocess.run(
        ["7z", "x", "-y", f"-o{EXTRACTED_DIR}", XML_7Z],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"Error: Extraction failed: {result.stderr.decode()}")
        return False

    # Fix permissions
    os.chmod(XML_FILE, 0o644)
    print("Extraction complete.")

    # Clean up 7z file to save space
    os.remove(XML_7Z)
    print("Removed 7z archive to save space.")

    return True


def sanitize_filename(title):
    """Create safe filename from page title."""
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe_title = safe_title.replace(' ', '_')
    safe_title = safe_title[:200]
    return f"{safe_title}.md"


def extract_pages():
    """Extract pages from MediaWiki XML dump."""
    # Check if already extracted
    if os.path.exists(PAGES_DIR):
        count = len([f for f in os.listdir(PAGES_DIR) if f.endswith('.md')])
        if count > 50000:
            print(f"Pages already extracted: {count} files in {PAGES_DIR}")
            return count

    os.makedirs(PAGES_DIR, exist_ok=True)

    extracted = 0
    skipped_ns = 0
    skipped_short = 0
    skipped_redirect = 0

    print(f"Parsing XML file: {XML_FILE}")
    print(f"Output directory: {PAGES_DIR}")

    context = ET.iterparse(XML_FILE, events=('end',))

    for event, elem in context:
        tag = elem.tag.replace('{http://www.mediawiki.org/xml/export-0.11/}', '')

        if tag == 'page':
            ns_elem = elem.find('mw:ns', NS)
            title_elem = elem.find('mw:title', NS)
            text_elem = elem.find('.//mw:text', NS)

            if ns_elem is None or title_elem is None:
                elem.clear()
                continue

            namespace = int(ns_elem.text) if ns_elem.text else -1
            title = title_elem.text if title_elem.text else ""
            text = text_elem.text if text_elem is not None and text_elem.text else ""

            if namespace != 0:
                skipped_ns += 1
                elem.clear()
                continue

            if text.strip().lower().startswith('#redirect'):
                skipped_redirect += 1
                elem.clear()
                continue

            if len(text.strip()) <= 200:
                skipped_short += 1
                elem.clear()
                continue

            filename = sanitize_filename(title)
            filepath = os.path.join(PAGES_DIR, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(text)

            extracted += 1
            if extracted % 5000 == 0:
                print(f"  Extracted {extracted} pages...")

            elem.clear()

    print(f"\nExtraction complete!")
    print(f"  Extracted: {extracted} pages")
    print(f"  Skipped (non-main namespace): {skipped_ns}")
    print(f"  Skipped (redirects): {skipped_redirect}")
    print(f"  Skipped (too short): {skipped_short}")

    return extracted


def main():
    print("=" * 50)
    print(f"{WIKI_NAME} Setup")
    print("=" * 50)
    print()

    check_dependencies()

    if not download_dump():
        sys.exit(1)

    extract_pages()

    print()
    print(f"Setup complete! You can now run:")
    print(f'  python generate_graphs.py --wiki-name "{WIKI_NAME}"')


if __name__ == "__main__":
    main()
