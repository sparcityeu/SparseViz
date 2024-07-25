#include "helpers.h"


vType calculateBin(vType coordinate, vType dimSize, vType numBins)
{
    if (dimSize == 0 || numBins == 0)
    {
        std::cerr << "Error: Division by zero in calculateBin function. dimSize: " << dimSize << ", numBins: " << numBins << std::endl;
        return -1;
    }

    vType binSize = dimSize / numBins; // integer division
    if (binSize == 0)
    {
        std::cerr << "Error: binSize is zero. dimSize: " << dimSize << ", numBins: " << numBins << std::endl;
        return -1;
    }

    vType bin = coordinate / binSize; // integer division

    if (bin >= numBins)
    {
        bin = numBins - 1;
    }

    if (bin < 0)
    {
        std::cerr << "Error: Negative bin index. coordinate: " << coordinate << ", binSize: " << binSize << ", dimSize: " << dimSize << ", numBins: " << numBins << std::endl;
        return -1;
    }

    return bin;
}


// helper function for escaping single quotes for later use in json prep.
std::string escapeSingleQuote(const std::string &input)
{
    std::string output;
    output.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        if (input[i] == '\'')
            output += "\\'";
        else
            output += input[i];
    }
    return output;
}


std::string calculateColorscale(bool is_exponential, bool is_red)
{
    std::string cscale;

    if (is_exponential)
    {
        if (is_red)
        {
            cscale = "[[0, 'rgb(255, 255, 255)'],"
                     "[0.00000000001, 'rgb(254, 189, 178)'],"
                     "[0.00000001, 'rgb(253, 170, 157)'],"
                     "[0.0000001, 'rgb(252, 151, 136)'],"
                     "[0.000001, 'rgb(251, 132, 115)'],"
                     "[0.00001, 'rgb(244, 109, 97)'],"
                     "[0.0001, 'rgb(237, 86, 79)'],"
                     "[0.001, 'rgb(230, 63, 61)'],"
                     "[0.01, 'rgb(224, 40, 40)'],"
                     "[0.1, 'rgb(217, 17, 17)'],"
                     "[1, 'rgb(210, 0, 0)']]";
        }
        else
        { // which means blue
            cscale = "[[0, 'rgb(255, 255, 255)'],"
                     "[ 0.000000000001, 'rgb(198, 219, 239)' ]," // approximating binary differentiton of colors
                     "[ 0.00000001, 'rgb(158, 202, 225)' ],"
                     "[ 0.0000001, 'rgb(107, 174, 214)' ],"
                     "[ 0.000001, 'rgb(66, 146, 198)' ],"
                     "[ 0.00001, 'rgb(33, 113, 181)' ],"
                     "[ 0.0001, 'rgb(8, 81, 156)' ],"
                     "[ 0.001, 'rgb(8, 48, 107)' ],"
                     "[ 0.01, 'rgb(7, 30, 80)' ],"
                     "[ 0.1, 'rgb(6, 20, 64)' ],"
                     "[ 1, 'rgb(3, 4, 30)' ] ] ";
        }
    }
    else
    {
        if (is_red)
        {
            cscale = "[[0, 'rgb(255, 255, 255)'],"
                     "[0.000000000001, 'rgb(254, 189, 178)'],"
                     "[0.1111111112, 'rgb(253, 170, 157)'],"
                     "[0.2222222223, 'rgb(252, 151, 136)'],"
                     "[0.3333333334, 'rgb(251, 132, 115)'],"
                     "[0.4444444445, 'rgb(244, 109, 97)'],"
                     "[0.5555555556, 'rgb(237, 86, 79)'],"
                     "[0.6666666667, 'rgb(230, 63, 61)'],"
                     "[0.7777777778, 'rgb(224, 40, 40)'],"
                     "[0.8888888889, 'rgb(217, 17, 17)'],"
                     "[1, 'rgb(210, 0, 0)']]";
        }
        else
        {
            cscale = "[[0, 'rgb(255, 255, 255)'],"
                     "[ 0.000000000001, 'rgb(198, 219, 239)' ],"
                     "[ 0.1111111112, 'rgb(158, 202, 225)' ],"
                     "[ 0.2222222223, 'rgb(107, 174, 214)' ],"
                     "[ 0.3333333334, 'rgb(66, 146, 198)' ],"
                     "[ 0.4444444445, 'rgb(33, 113, 181)' ],"
                     "[ 0.5555555556, 'rgb(8, 81, 156)' ],"
                     "[ 0.6666666667, 'rgb(8, 48, 107)' ],"
                     "[ 0.7777777778, 'rgb(7, 30, 80)' ],"
                     "[ 0.8888888889, 'rgb(6, 20, 64)' ],"
                     "[ 1, 'rgb(3, 4, 30)' ] ] ";
        }
    }
    return cscale;
}


void trim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
    {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
    {
        return !std::isspace(ch);
    }).base(), s.end());
}


std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter))
    {
        trim(token);
        tokens.push_back(token);
    }

    return tokens;
}


std::string merge(const std::vector<std::string>& splittedVersion)
{
    std::string merged;
    for (const auto& token: splittedVersion)
    {
        merged += token;
    }
    return merged;
}


std::string lowerString(const std::string& s)
{
    std::string loweredString;
    for (const auto& c: s)
    {
        loweredString += tolower(c);
    }
    return loweredString;
}


std::string getLine(char*& addr)
{
    std::string line;
    while ((*addr) != '\n')
    {
        line += (*addr);
        ++addr;
    }
    ++addr;
    return line;
}
