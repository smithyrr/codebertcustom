import requests
from bs4 import BeautifulSoup
import json

url = "https://community.bistudio.com/wiki/Category:Arma_3:_Scripting_Commands"
base_url = "https://community.bistudio.com"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")
commands_list = soup.find_all("div", class_="mw-category-group")

commands_data = []

for group in commands_list:
    commands = group.find_all("a")
    for command in commands:
        command_url = base_url + command["href"]
        command_response = requests.get(command_url)
        command_soup = BeautifulSoup(command_response.text, "html.parser")
        
        command_name = command.text.strip()
        command_description = command_soup.find("div", class_="description").get_text(strip=True) if command_soup.find("div", class_="description") else "No description available."
        command_example = command_soup.find("pre", class_="code-snippet-example").get_text(strip=True) if command_soup.find("pre", class_="code-snippet-example") else "No example available."
        
        commands_data.append({
            "name": command_name,
            "description": command_description,
            "example": command_example
        })

with open("arma3_commands.json", "w") as file:
    json.dump(commands_data, file, indent=4)
