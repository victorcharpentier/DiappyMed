import requests
from bs4 import BeautifulSoup
import pandas as pd

pages = list()
for i in range(1, 5):
    url = 'https://web.archive.org/web/20121007172955/https://www.nga.gov/collection/anZ' + \
          str(i) + '.htm'
    pages.append(url)

names = list()
links = list()

for item in pages:
    page = requests.get(item)
    soup = BeautifulSoup(page.text, 'html.parser')

    # Remove bottom links
    last_links = soup.find(attrs={'class': 'AlphaNav'})
    last_links.decompose()

    # Pull all text from the BodyText div
    artist_name_list = soup.find(attrs={'class': 'BodyText'})

    # Pull text from all instances of <a> tag within BodyText div
    artist_name_list_items = artist_name_list.find_all('a')

    # Create for loop to print out all artists' names
    # for artist_name in artist_name_list_items:
    # print(artist_name.prettify())

    names = list()
    links = list()

    # Use .contents to pull out the <a> tag’s children
    for artist_name in artist_name_list_items:
        link = 'https://web.archive.org' + artist_name.get('href')
        fullname = artist_name.contents[0]
        names.append(fullname)
        links.append(link)

# We pass a dictionary where each list is associated with a column name
df = pd.DataFrame({'name': names, 'link': links})

# We write to a file:
df.to_csv("myfile.csv")