import requests
import os
from bs4 import BeautifulSoup

# Define the base URLs of the websites
bohemia_url = "https://community.bistudio.com"
other_url = "https://dayzexplorer.zeroy.com"

# Define a function to download a code example and save it to a file
def download_code_example(url, subdir, example_num):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the HTML "code" tag that contains the example code
    code_tag = soup.find("code")
    if code_tag is not None:
        # Extract the example code and save it to a file
        code_example = code_tag.get_text()
        filename = f"{subdir}_example{example_num}.sqf"
        with open(filename, "w") as f:
            f.write(code_example)

# Get the HTML content of the "Category:Arma_3:_Scripting_Commands" page on the Bohemia Interactive Community Wiki
response = requests.get(bohemia_url + "/wiki/Category:Arma_3:_Scripting_Commands")
soup = BeautifulSoup(response.content, "html.parser")

# Find the HTML "div" tag that contains the list of sub-directories
subdir_div = soup.find("div", {"id": "mw-pages"})

# Find all HTML "a" tags within the "div" tag and extract their contents and URLs
subdir_links = subdir_div.find_all("a")
for link in subdir_links:
    subdir_name = link.get_text()
    subdir_url = bohemia_url + link["href"]
    subdir_response = requests.get(subdir_url)
    subdir_soup = BeautifulSoup(subdir_response.content, "html.parser")
    # Find all HTML "a" tags within the sub-directory and extract their contents and URLs
    example_links = subdir_soup.find_all("a", {"class": "mw-redirect"})
    for i, link in enumerate(example_links):
        example_name = link.get_text()
        example_url = bohemia_url + link["href"]
        # Download and save the code example to a file
        download_code_example(example_url, subdir_name, i)

# Get the HTML content of the DayZ Explorer website
response = requests.get(other_url)
soup = BeautifulSoup(response.content, "html.parser")

# Find all HTML "code" tags within the website and extract their contents
code_tags = soup.find_all("code")
for i, tag in enumerate(code_tags):
    code_example = tag.get_text()
    # Save the code example to a file
    filename = f"dayz_example{i}.txt"
    with open(filename, "w") as f:
        f.write(code_example)
