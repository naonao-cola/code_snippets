#include <iostream>
#include <fstream>
#include <string>
#include "addressbook.pb.h"

void PromptForAddress(tutorial::Person* person) {

    std::cout << "Enter person ID number: ";
    int id;
    std::cin >> id;
    person->set_id(id);
    std::cin.ignore(256, '\n');

    std::cout << "Enter name: ";
    getline(std::cin, *person->mutable_name());

    std::cout << "Enter email address (blank for none): ";
    std::string email;
    getline(std::cin, email);
    if (!email.empty()) {
        person->set_email(email);
    }

    while (true) {
    std::cout << "Enter a phone number (or leave blank to finish): ";
    std::string number;
    getline(std::cin, number);
    if (number.empty()) {
      break;
    }

    tutorial::Person::PhoneNumber* phone_number = person->add_phones();
    phone_number->set_number(number);

    std::cout << "Is this a mobile, home, or work phone? ";
    std::string type;
    getline(std::cin, type);
    if (type == "mobile") {
      phone_number->set_type(tutorial::Person::MOBILE);
    } else if (type == "home") {
      phone_number->set_type(tutorial::Person::HOME);
    } else if (type == "work") {
      phone_number->set_type(tutorial::Person::WORK);
    } else {
      std::cout << "Unknown phone type.  Using default." << std::endl;
    }
  }
}
int main(int argc, char** argv) {
    std::cout << "hello world!" << std::endl;

    // tutorial::AddressBook address_book;
    // address_book.add_people();

    tutorial::Person p;
    p.set_id(1);
    p.set_name("yang");
    p.set_email("yang@gmail.com");


    // 序列化
    std::string buffer;
    p.SerializeToString(&buffer);

    // 反序列化
    tutorial::Person person2;
    person2.ParseFromString(buffer);
    std::cout << "ID: " << person2.id() << " Name: " << person2.name() << std::endl;
    
    return 0;
}
