#ifndef ARGS_PARSER_HPP
#define ARGS_PARSER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace input {

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class parser {
    public:
        parser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        /// @author iain
        const std::string& get(const std::string &option) const{
            std::string option_with_prefix = "--" + option;
            std::vector<std::string>::const_iterator itr;
            itr = std::find(this->tokens.begin(), this->tokens.end(), option_with_prefix);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

        /// @author iain
        bool exists(const std::string &option) const{
            std::string option_with_prefix = "--" + option;
            return std::find(this->tokens.begin(), this->tokens.end(), option_with_prefix)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

};

#endif // ARGS_PARSER_HPP