import requests
import json

# URL for the POST request
url = 'https://www.interactivebrokers.com/webrest/search/products-by-filters'

# Headers including Content-Type and User-Agent
headers = {
    'Content-Type': 'application/json;charset=UTF-8',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0',
    'Cookie': 'x-sess-uuid=0.4f521302.1721295570.797204;'  # Example, replace with your current session cookie
}



import requests
import pandas as pd
import json



# Initialize an empty DataFrame to store all products
all_products = pd.DataFrame()

# Loop through all page numbers
for page in range(1, 723):  # Adjust the range as necessary
    print(page)
    # Body data for the POST request
    data = {
        "pageNumber": page,
        "pageSize": "100",
        "sortField": "symbol",
        "sortDirection": "asc",
        "productCountry": ["US"],
        "productSymbol": "",
        "newProduct": "all",
        "productType": ["OPT"],
        "domain": "com"
    }

    # Convert the dictionary to JSON format
    json_data = json.dumps(data)

    # Making the POST request
    response = requests.post(url, headers=headers, data=json_data)

    # Checking the response from the server
    if response.status_code == 200:
        response_data = response.json()
        # Check if 'products' is in the response data
        if 'products' in response_data:
            # Convert the list of products to a DataFrame
            current_page_products = pd.DataFrame(response_data['products'])
            # Append the current page's products to the all_products DataFrame
            all_products = pd.concat([all_products, current_page_products], ignore_index=True)
    else:
        print(f"Failed to fetch data for page {page}: {response.status_code}")

# Save the DataFrame to a CSV file after all pages have been processed
all_products.to_csv('products_data.csv', index=False)
print("All product data has been fetched and saved to 'products_data.csv'.")
