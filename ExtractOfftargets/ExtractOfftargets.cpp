#include "ExtractOfftargets.hpp"

using std::vector;
using std::string;
using std::fstream;
using std::ofstream;
using std::ifstream;
using boost::algorithm::trim;
using boost::algorithm::to_upper;
using boost::algorithm::split;
using boost::regex;
using boost::sregex_iterator;
using boost::smatch;
namespace fs = std::filesystem;

const regex fwdExp = regex("(?=([ACG][ACGT]{19}[ACGT][AG]G))");
const regex bwdExp = regex("(?=(C[CT][ACGT][ACGT]{19}[TGC]))");

int main(int argc, char** argv)
{
    // Check number of args
    if (argc < 3)
    {
        std::cerr << fmt::format("Usage: {} <output-file>  {{<input-file-1> <input-file-2> ... <input-file-n> | <input-dir>}}\n", argv[0]) << std::endl;
        exit(1);
    }

    auto startTime = std::chrono::steady_clock::now();
    fs::path tempWorkingDir = fs::temp_directory_path() / "Crackling-extractOfftargets";
    fs::create_directory(tempWorkingDir);
    std::atomic_ullong inputFileCounter = 0;
    vector<string> filesToProcess;

    for (int i = 2; i < argc; i++)
    {
        // Command line input
        fs::path input(argv[i]);

        // File 
        if (fs::is_regular_file(input))
        {
            filesToProcess.push_back(input.string());
        }
        // Directory
        else if (fs::is_directory(input))
        {
            for (const fs::path& file : fs::directory_iterator(input))
            {
                filesToProcess.push_back(file.string());
            }
        }
        // Skip otherwise
        else
        {
            string errorMsg = fmt::format("Error processing: {}\nSkipping entry, Please check the path and try again.", argv[i]);
            std::cout << errorMsg << std::endl;
            continue;
        }
    }

    std::cout << "Spliting input(s)" << std::endl;
    // Split each sequence to it's own file
    // #pragma omp parallel for schedule(static,1)
    for (const string& file : filesToProcess)
    {
        ofstream tempOutFile;
        ifstream inFile;
        string inputLine;
        inFile.open(file, std::ios::binary | std::ios::in);
        std::getline(inFile, inputLine);
        // file is Fasta formatted
        if (inputLine[0] == '>')
        {
            // Remove all line breaks between sequences segments
            tempOutFile.open((tempWorkingDir / fmt::format("{}.txt", inputFileCounter++)).string(), std::ios::binary | std::ios::out);
            while (std::getline(inFile, inputLine))
            {
                trim(inputLine);
                to_upper(inputLine);
                if (inputLine[0] == '>')
                {
                    tempOutFile.close();
                    tempOutFile.open((tempWorkingDir / fmt::format("{}.txt", inputFileCounter++)).string(), std::ios::binary | std::ios::out);
                }
                else
                {
                    tempOutFile << inputLine;
                }
            }
        }
        // file is plain text, assume one sequence per line
        else
        {
            do
            {
                tempOutFile.open((tempWorkingDir / fmt::format("{}.txt", inputFileCounter++)).string(), std::ios::binary | std::ios::out);
                trim(inputLine);
                to_upper(inputLine);
                tempOutFile << inputLine;
                tempOutFile.close();
            } while (std::getline(inFile, inputLine));
        }
        tempOutFile.close();
        inFile.close();
    }

    std::cout << "Done" << std::endl;

    std::cout << "Identifying off-targets" << std::endl;
    // Mulithread process each extacts off targets
    std::atomic_ullong batchFileCounter = 0;
    uint64_t offTargetBatchSize = 30000000;
    // #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < inputFileCounter; i++)
    {
        ofstream outFile;
        ifstream inFile;
        string inputLine;
        uint64_t offtargetsFound = 0;
        inFile.open((tempWorkingDir / fmt::format("{}.txt", i)).string(), std::ios::binary | std::ios::in);
        outFile.open((tempWorkingDir / fmt::format("{}_batch.txt", batchFileCounter++)).string(), std::ios::binary | std::ios::out);
        for (inputLine; std::getline(inFile, inputLine);)
        {
            // std::getline(inFile, inputLine);
            trim(inputLine);
            // Add forward matches
            for (sregex_iterator regexItr(inputLine.begin(), inputLine.end(), fwdExp); regexItr != sregex_iterator(); regexItr++)
            {
                smatch m = *regexItr;
                if (offtargetsFound < offTargetBatchSize)
                {
                    outFile << m[1].str().substr(0, 20) << "\n";
                    ++offtargetsFound;
                }
                else 
                {
                    outFile.close();
                    outFile.open((tempWorkingDir / fmt::format("{}_batch.txt", batchFileCounter++)).string(), std::ios::binary | std::ios::out);
                    outFile << m[1].str().substr(0, 20) << "\n";
                    offtargetsFound = 1;
                }
            }
            // Add reverse matches
            for (sregex_iterator regexItr(inputLine.begin(), inputLine.end(), bwdExp); regexItr != sregex_iterator(); regexItr++)
            {
                smatch m = *regexItr;
                if (offtargetsFound < offTargetBatchSize)
                {
                    outFile << rc(m[1].str()).substr(0, 20) << "\n";
                    ++offtargetsFound;
                }
                else 
                {
                    outFile.close();
                    outFile.open((tempWorkingDir / fmt::format("{}_batch.txt", batchFileCounter++)).string(), std::ios::binary | std::ios::out);
                    outFile << rc(m[1].str()).substr(0, 20) << "\n";
                    offtargetsFound = 1;
                }
            }
        }
        inFile.close();
        outFile.close();
        fs::remove((tempWorkingDir / fmt::format("{}.txt", i)).string());
    }
    std::cout << "Done" << std::endl;

    std::cout << "Sorting batch files" << std::endl;
    // #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < batchFileCounter; i++)
    {

        std::vector<string> offTargets;
        ifstream inFile;
        ofstream outFile;
        string inputLine;
        inFile.open((tempWorkingDir / fmt::format("{}_batch.txt", i)).string(), std::ios::binary | std::ios::in);
        offTargets.reserve(offTargetBatchSize);
        while(std::getline(inFile, inputLine))
        {
            trim(inputLine);
            offTargets.push_back(inputLine);
        }
        inFile.close();
        std::sort(offTargets.begin(), offTargets.end());
        outFile.open((tempWorkingDir / fmt::format("{}_sorted.txt", i)).string(), std::ios::binary | std::ios::out);
        for (const string& s : offTargets)
        {
            outFile << s << "\n";
        }
        outFile.close();
        fs::remove((tempWorkingDir / fmt::format("{}_batch.txt", i)).string());
    }
    std::cout << "Done" << std::endl;

    std::cout << "Merging (Parrallel)" << std::endl;
    // Merge sorted files
    int threads = 16;
    vector<string> filesToMerge;
    if (batchFileCounter < threads * 2)
    {
        for (int i = 0; i < batchFileCounter; ++i)
        {
            filesToMerge.push_back((tempWorkingDir / fmt::format("{}_sorted.txt", i)).string());
        }
    }
    else
    {
        for (int i = 0; i < threads; ++i)
        {
            filesToMerge.push_back((tempWorkingDir / fmt::format("{}_merged.txt", i)).string());
        }
        int batchSize = ceil(batchFileCounter / (threads * 1.0));

        // #pragma omp parallel for schedule(static,1)
        for (int i = 0; i < threads; ++i)
        {

            vector<string> fileNames;
            for (int j = i; j < batchFileCounter; j += threads)
            {
                fileNames.push_back((tempWorkingDir / fmt::format("{}_sorted.txt", j)).string());
            }

            vector<ifstream> sortedFiles(fileNames.size());
            vector<string> offTargets(fileNames.size());
            for (int j = 0; j < fileNames.size(); ++j)
            {
                sortedFiles[j].open(fileNames[j], std::ios::binary | std::ios::in);
                std::getline(sortedFiles[j], offTargets[j]);
            }

            // Remove any empty files
            auto sortedFilesIter = sortedFiles.begin();
            auto offTargetsIter = offTargets.begin();
            auto fileNamesIter = fileNames.begin();
            while (sortedFilesIter != sortedFiles.end())
            {
                if (sortedFilesIter->eof())
                {
                    sortedFilesIter = sortedFiles.erase(sortedFilesIter);
                    offTargetsIter = offTargets.erase(offTargetsIter);
                    fs::remove(*fileNamesIter);
                    fileNamesIter = fileNames.erase(fileNamesIter);
                }
                else
                {
                    ++sortedFilesIter;
                    ++offTargetsIter;
                    ++fileNamesIter;
                }
            }

            ofstream mergedFile;
            mergedFile.open((tempWorkingDir / fmt::format("{}_merged.txt", i)).string(), std::ios::binary | std::ios::out);
            while (sortedFiles.size() > 1)
            {
                // Find index of lowest off-target
                int lowest = 0;
                for (int j = 0; j < sortedFiles.size(); ++j)
                {
                    if (offTargets[j] < offTargets[lowest])
                    {
                        lowest = j;
                    }
                }
                // Write to file
                mergedFile << offTargets[lowest] << "\n";
                // Update offtargets
                std::getline(sortedFiles[lowest], offTargets[lowest]);
                // If at EOF remove from list
                if (sortedFiles[lowest].eof())
                {
                    sortedFiles[lowest].close();
                    fs::remove(fileNames[lowest]);
                    sortedFiles.erase(sortedFiles.begin() + lowest);
                    offTargets.erase(offTargets.begin() + lowest);
                    fileNames.erase(fileNames.begin() + lowest);
                }
            }

            mergedFile << offTargets[0] << "\n";
            while (std::getline(sortedFiles[0], offTargets[0]))
            {
                mergedFile << offTargets[0] << "\n";
            }
            sortedFiles[0].close();
            fs::remove(fileNames[0]);
            mergedFile.close();
        }
    }
    std::cout << "Done" << std::endl;

    std::cout << "Merging (Serial)" << std::endl;

    vector<ifstream> mergedFiles(filesToMerge.size());
    vector<string> offTargets(filesToMerge.size());
    for (int i = 0; i < filesToMerge.size(); i++)
    {
        mergedFiles[i].open(filesToMerge[i], std::ios::binary | std::ios::in);
        std::getline(mergedFiles[i], offTargets[i]);
    }

    // Remove any empty files
    auto mergedFilesIter = mergedFiles.begin();
    auto offTargetsIter = offTargets.begin();
    while (mergedFilesIter != mergedFiles.end())
    {
        if (mergedFilesIter->eof())
        {
            mergedFilesIter = mergedFiles.erase(mergedFilesIter);
            offTargetsIter = offTargets.erase(offTargetsIter);
        }
        else
        {
            ++mergedFilesIter;
            ++offTargetsIter;
        }
    }

    ofstream finalOutput;
    finalOutput.open(argv[1], std::ios::binary | std::ios::out);
    while (mergedFiles.size() > 1)
    {
        // Find index of lowest off-target
        int lowest = 0;
        for (int j = 0; j < mergedFiles.size(); j++)
        {
            if (offTargets[j] < offTargets[lowest])
            {
                lowest = j;
            }
        }
        // Write to file
        finalOutput << offTargets[lowest] << "\n";
        // Update offtargets
        std::getline(mergedFiles[lowest], offTargets[lowest]);
        // If at EOF remove from list
        if (mergedFiles[lowest].eof())
        {
            mergedFiles[lowest].close();
            fs::remove(filesToMerge[lowest]);
            mergedFiles.erase(mergedFiles.begin() + lowest);
            offTargets.erase(offTargets.begin() + lowest);
            filesToMerge.erase(filesToMerge.begin() + lowest);
        }
    }

    finalOutput << offTargets[0] << "\n";
    while (std::getline(mergedFiles[0], offTargets[0]))
    {
        finalOutput << offTargets[0] << "\n";
    }
    mergedFiles[0].close();
    finalOutput.close();

    std::cout << "Done" << std::endl;

    std::cout << "Cleaning intermediate files" << std::endl;
    fs::remove_all(tempWorkingDir);
    std::cout << "Done" << std::endl;

    // Report time taken (total)
    std::chrono::nanoseconds nanoSec = std::chrono::steady_clock::now() - startTime;
    std::chrono::duration<uint64_t, std::ratio<86400>> days = std::chrono::duration_cast<std::chrono::duration<uint64_t, std::ratio<86400>>>(nanoSec);
    std::chrono::hours hours = std::chrono::duration_cast<std::chrono::hours>(nanoSec - days);
    std::chrono::minutes minutes = std::chrono::duration_cast<std::chrono::minutes>(nanoSec - days - hours);
    std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(nanoSec - days - hours - minutes);
    std::chrono::milliseconds milliSec = std::chrono::duration_cast<std::chrono::milliseconds>(nanoSec - days - hours - minutes - sec);
    std::chrono::microseconds microSec = std::chrono::duration_cast<std::chrono::microseconds>(nanoSec - days - hours - minutes - sec - milliSec);
    std::cout << fmt::format("Total run time {:02} {:02}:{:02}:{:02} (dd hh:mm:ss) or {} seconds", days.count(), hours.count(), minutes.count(), sec.count(), std::chrono::duration_cast<std::chrono::seconds>(nanoSec).count()) << std::endl;
}