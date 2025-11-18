/**
 * @FilePath     : /lua_test/src/main.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2025-11-18 13:10:50
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2025-11-18 13:56:29
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#include <iostream>
#include <sol/sol.hpp>

/**

https://github.com/ThisisGame/cpp-game-engine-book

https://www.thisisgame.com.cn/tutorial?book=cpp-game-engine-book&lang=zh&md=17.%20integrate_lua/17.1%20sol2_interaction_with_cpp.md


https://blog.csdn.net/weixin_42849849/article/details/153524980

*/
int test_01() {
  // 创建 Lua 状态机
  sol::state lua;

  // 打开标准库
  lua.open_libraries(sol::lib::base, sol::lib::package);

  // 执行 Lua 代码
  lua.script("print('Hello from Lua!')");
  return 0;
}

int add(int a, int b) { return a + b; }

void greet(const std::string &name) { std::cout << "Hello, " << name << "!\n"; }

int test_02() {
  sol::state lua;
  lua.open_libraries(sol::lib::base);
  // 绑定 C++ 函数到 Lua
  lua.set_function("add", add);
  lua.set_function("greet", greet);
  // 在 Lua 中调用绑定的函数
  lua.script(R"(
        result = add(5, 7)
        print("The sum is: " .. result)
        greet("World")
    )");

  return 0;
}

class Player {
private:
  std::string name;
  int health;

public:
  Player(const std::string &n, int h) : name(n), health(h) {}

  void take_damage(int damage) { health = std::max(0, health - damage); }

  void heal(int amount) { health += amount; }

  std::string get_name() const { return name; }
  int get_health() const { return health; }
};

int test_03() {

  sol::state lua;
  lua.open_libraries();

  // 绑定类
  lua.new_usertype<Player>(
      "Player", "new", sol::constructors<Player(const std::string &, int)>(),
      "take_damage", &Player::take_damage, "heal", &Player::heal, "get_name",
      &Player::get_name, "get_health", &Player::get_health, "name",
      sol::property(&Player::get_name), // 属性访问
      "health", sol::property(&Player::get_health));

  lua.script(R"(
        player = Player.new("Hero", 100)
        print("Player:", player.name, "Health:", player.health)
        player:take_damage(30)
        print("After damage:", player.health)
        player:heal(10)
        print("After heal:", player.health)
    )");
  return 0;
}

int main(int argc, char **argv) {
  std::cout << "hello world!" << std::endl;
  //   test_01();
  //   test_02();
  test_03();
  return 0;
}
