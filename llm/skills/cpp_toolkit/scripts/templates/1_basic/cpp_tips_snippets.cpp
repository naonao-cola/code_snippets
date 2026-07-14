#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <array>

// ============================================================================
// 1. 智能指针高级用法 (Smart Pointers)
// ============================================================================
class Student {
public:
	Student() {};
	Student(int id, int grades, std::string name) :ID(id), Grades(grades), Name(name) {};
	~Student() {};
	int ID;
	int Grades;
	std::string Name;
};

void Delete(Student* s) { delete[] s; }

void smart_pointer_examples() {
	// 1.1 shared_ptr 配合不同 Deleter
	std::shared_ptr<Student> student1(new Student(1, 91, "nity_noe"), [](Student* s){ delete s; });
	std::shared_ptr<Student> student2(new Student[5], [](Student* s) { delete[] s; });
	std::shared_ptr<Student> student3(new Student[5], std::default_delete<Student[]>());
	std::shared_ptr<Student> student4(new Student[5], Delete);
	
	// 1.2 二维指针数组 (第一维动态，第二维静态)
	std::shared_ptr<Student[1024][10]> pt(new Student[1024][10], [=](Student(*p)[10])->void { delete[] p; });
	pt[1023][20].ID = 5; // 注意：shared_ptr 的数组没有 [] 重载，但在某些高版本或特定声明下可通过 get() 访问
	
	// 1.3 unique_ptr 数组用法
	std::unique_ptr<int[]> pnColhistT(new int[100] {0});
	auto ptr2 = std::make_unique<std::array<int, 5>>(std::array<int, 5>{1, 2, 3, 4, 5});
}

// 1.4 妙用：将 shared_ptr 作为互斥锁的自动释放器
class Mutex { public: void lock() {} void unlock() {} };
void unlock(Mutex* m) { m->unlock(); }
void lock(Mutex* m) { m->lock(); }

class Lock {
public:
	explicit Lock(Mutex* pm) : mutexPtr(pm, unlock) {
		lock(mutexPtr.get());
	}
private:
	std::shared_ptr<Mutex> mutexPtr;
};

// ============================================================================
// 2. 编译依存性降低 (Handle & Interface Classes - Effective C++ 条款 31)
// ============================================================================
// Person.h (Interface Class)
class MyAddress;
class MyDate;
class Person {
public:
    static Person* CreatePerson(const std::string &name, const MyAddress& addr, const MyDate& date);
    virtual std::string GetName() const = 0;
    virtual ~Person(){}
};

// RealPerson.h (Implementation)
class RealPerson: public Person {
private:
    std::string Name;
public:
    RealPerson(std::string name) : Name(name) {}
    virtual std::string GetName() const override { return Name; }
};

// RealPerson.cpp
Person* Person::CreatePerson(const std::string& name, const MyAddress& addr, const MyDate& date) {
    // 隐藏了 RealPerson 的实现细节，对外仅暴露 Person 接口
    return new RealPerson(name);
}

// ============================================================================
// 3. 强制类型转换 (Casting)
// ============================================================================
class CastA {
public:
    int i = 5;
    const std::string m_s = "Test String.";
    operator int() { return 1; }
};

void cast_examples() {
    CastA a;
    
    // 3.1 static_cast (低风险转换)
    int n = static_cast<int>(3.14); // 3
    int n2 = static_cast<int>(a);   // 触发 operator int()
    
    // 3.2 reinterpret_cast (高风险比特拷贝)
    long long la = 0x12345678abcdLL;
    CastA* pa = reinterpret_cast<CastA*>(la);
    // pa->i = 400; // 非常危险，极可能崩溃
    
    // 3.3 const_cast (去除 const 属性)
    std::string& p_str = const_cast<std::string&>(a.m_s);
    p_str = "New Test String!";
    
    // 3.4 dynamic_cast (多态安全向下转型)
    // 需包含虚函数的基类指针转换，失败返回 NULL
}