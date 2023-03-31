import json

data = json.load(open("arma3_commands_with_descriptions.json"))

with open("formatted_arma3_commands_with_descriptions.json", "w") as f:
    for item in data:
        for snippet in item["snippets"]:
            f.write(json.dumps({
                "code_snippet": snippet["code"],
                "language": "Arma 3",
                "metadata": {
                    "name": item["name"],
                    "description": item["description"],
                    "example": snippet["example"]
                }
            }) + "\n")
