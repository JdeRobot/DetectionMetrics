//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <SampleGeneratorLib/Logger.h>
#include <highgui.h>
#include <SampleGeneratorLib/DatasetConverters/OwnDatasetReader.h>
#include <SampleGeneratorLib/DatasetConverters/YoloDatasetReader.h>
#include <SampleGeneratorLib/DatasetConverters/SpinelloDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DarknetInferencer.h>
#include <FrameworkEvaluator/MassInferencer.h>


namespace
{
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t SUCCESS = 0;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;

} // namespace





struct EvaluatorArguments{
    std::string inputPath;
    std::string outputPath;

};


int parse_arguments(const int argc, char* argv[], EvaluatorArguments& args){
    try
    {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("input,i", po::value<std::string>(&args.inputPath)->required())
                ("output,o", po::value<std::string>(&args.outputPath)->required());

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc),
                      vm); // can throw

            /** --help option
             */
            if ( vm.count("help")  )
            {
                std::cout << "Basic Command Line Parameter App" << std::endl
                          << desc << std::endl;
                return SUCCESS;
            }

            po::notify(vm); // throws on error, so do after help in case
            // there are any problems
        }
        catch(po::error& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }

        // application code here //

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
                  << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;

    }
    return SUCCESS;

}





int main (int argc, char* argv[]) {

    EvaluatorArguments args;
    if (parse_arguments(argc,argv,args) != SUCCESS){
        std::cout << "error" << std::endl;
        return(1);
    }


    Logger::getInstance()->setLevel(Logger::INFO);
    Logger::getInstance()->info("Reviewing " + args.inputPath);


//    OwnDatasetReaderPtr reader( new OwnDatasetReader(args.inputPath));
//    YoloDatasetReader reader(args.path);
    YoloDatasetReaderPtr reader( new YoloDatasetReader(args.inputPath));
    DarknetInferencerPtr evaluator(new DarknetInferencer("/home/frivas/devel/external/darknet/cfg/yolo-voc.cfg", "/home/frivas/devel/external/darknet/yolo-voc.weights"));

    MassInferencer inferencer(reader,evaluator,args.outputPath, true);
    inferencer.process();
}
