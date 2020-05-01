from bs4 import BeautifulSoup
import csv
import sys


"""Extract name/messenger and phone/tag from quiz result comtact list"""
def get_contacts(file_path_ntml):
    with open(file_path_ntml, 'r') as f:
        content = f.read()
        output_list = []
        soup = BeautifulSoup(content, 'lxml')
        all_select = soup.select('body > div:nth-child(1) > div > div.app-body.app__body > main > div > div > div > div > div.card.leads > div.card-body > div > table > tbody > tr > td')
        account = [item for item in all_select if item.select(".mdi-phone")]
        messenger = [item for item in all_select if item.select(".leads__messenger-icon")]
        print("Find contacts:", len(account) + len(messenger))
        for a in account:
            all_div = a.findAll("div")
            name = all_div[0].getText()
            phone = all_div[1].getText()
            output_list.append([name, phone])
            print(name, phone)

        for m in messenger:
            image = m.div.img['src']
            app_name = "---not recognize---"
            if str(image).find("telegram") != -1:
                app_name = "telegram"
            elif str(image).find("PHN2ZyBpZD0iaW5zdGFncmFtXzFfIi") != -1:
                app_name = "instagram"
            elif str(image).find("viber") != -1:
                app_name = "viber"
            elif str(image).find("whatsapp") != -1:
                app_name = "whatsapp"
            contact = m.div.getText()
            output_list.append([app_name, contact])
            print(app_name, contact)

        return output_list

def write_csv(content, header = ["name/messanger", "contact"], file_path="contact_base.csv"):
    output_file = open(file_path, "w")
    with output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        for row in content:
            writer.writerow(row)

if __name__ == "__main__":
    print("'python marquiz_contacts_parser.py <path_to_html> <path_to_output_csv>'\n")
    path_to_html = "Site-Tel/Marquiz - конструктор маркетинговых квизов.html"
    path_to_csv = "contact_base.csv"
    if len(sys.argv) > 1:
        path_to_html = sys.argv[1]
    if len(sys.argv) > 2:
        path_to_csv = sys.argv[2]
    print("Input file:", path_to_html)
    print("Output file:", path_to_csv)
    print()
    contacts = get_contacts(path_to_html)
    write_csv(contacts, file_path = path_to_csv)
