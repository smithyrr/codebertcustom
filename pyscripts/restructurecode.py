import json

# Read the content from the original JSON file
with open("arma3_commands_by_functionality.json", "r") as file:
    content = file.read()

# Remove the leading and trailing commas from the content
content = content.strip(", ")

# Wrap the content in square brackets to form a JSON array
formatted_content = f"[{content}]"

# Parse the formatted content as a JSON object
formatted_data = json.loads(formatted_content)

# Save the formatted content to a new JSON file
with open("formatted_arma3_commands_by_functionality.json", "w") as file:
    json.dump(formatted_data, file, indent=4)
