#ifndef PLAYER_H
#define PLAYER_H

#include <string>
#include <vector>
using namespace std;

class Player {

private:
    string name;
    int money;
    // Add any other player attributes here ( position on the board, properties owned, etc.)
    vector<string> propertiesOwned;
    string currentPosition;

public:
    // Constructor
    Player(const std::string& playerName, int startingMoney, string startingPosition);

    // Getters
    string getName() const;
    int getMoney() const;
    string getCurrentPosition() const;

    // Setters / Game Logic
    void addMoney(int amount);
    void deductMoney(int amount);
    void setMoney(int newAmount);
    void setCurrentPosition(const string& newPosition);
};

#endif //PLAYER_H
