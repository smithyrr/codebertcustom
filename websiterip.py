import requests
from bs4 import BeautifulSoup

# Download the webpage content
url = "https://community.bistudio.com/wiki/DayZ:Enforce_Script_Syntax"
response = requests.get(url)
html_content = response.content

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Extract the text content from the webpage
text_content = soup.get_text()

# Print the result
print(text_content)
