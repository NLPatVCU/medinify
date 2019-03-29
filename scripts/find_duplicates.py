import csv

urls = []
duplicates = []

with open('heart-urls.csv', 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:
        urls.append(row['URL'])

for url in urls:
    url_count = 0

    for u in urls:
        if url == u:
            url_count += 1

    if url_count > 1 and url not in duplicates:
        duplicates.append(url)

for dupe in duplicates:
    print(dupe)