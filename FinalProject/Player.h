#ifndef PLAYER_H
#define PLAYER_H

#include <string>
using namespace std;

class Player {

private:
    string name;
    int money;

public:
    // Constructor
    Player(const std::string& playerName, int startingMoney);

    // Getters
    string getName() const;
    int getMoney() const;

    // Setters / Game Logic
    void addMoney(int amount);
    void deductMoney(int amount);
    void setMoney(int newAmount);
};

#endif //PLAYER_H
