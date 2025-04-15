// Player.cpp
#include "Player.h"

// Constructor
Player::Player(const std::string& playerName, int startingMoney, string startingPosition)
: name(playerName), money(startingMoney), currentPosition(startingPosition) {}

//-----------------------------------------------------------------------
// Getters
std::string Player::getName() const
{
    return name;
}
int Player::getMoney() const
{
    return money;
}

string Player::getCurrentPosition() const
{
    return currentPosition;
}

//-----------------------------------------------------------------------

// Setters / Game Logic
void Player::addMoney(int amount)
{
    money += amount;
}
void Player::deductMoney(int amount)
{
    money -= amount;
}
void Player::setMoney(int newAmount)
{
    money = newAmount;
}

void Player::setCurrentPosition(const string& newPosition)
{
    currentPosition = newPosition;
}
