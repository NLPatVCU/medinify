drugs = []

with open('cancer-drug-names-original.txt') as file:
    for line in file:
        drug = line.strip()

        if '(' in drug:
            split = drug.split('(')
            name = split[0]
            generic = split[1].strip('()')
            drugs.append(name)
            drugs.append(generic)
        else:
            drugs.append(drug)

for drug in drugs:
    print(drug)
