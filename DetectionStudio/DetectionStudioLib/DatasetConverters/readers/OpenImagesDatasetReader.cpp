#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "OpenImagesDatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

#include <istream>
#include <string>
#include <vector>


using namespace boost::filesystem;



OpenImagesDatasetReader::OpenImagesDatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}


bool OpenImagesDatasetReader::find_img_directory(const path & dir_path, path & path_found, std::string& img_dirname) {
    LOG(INFO) << dir_path.string() << " " << img_dirname << '\n';

    directory_iterator end_itr;
    int count = 0;
    for (directory_iterator itr( dir_path ); itr != end_itr; ++itr) {
        if (is_directory(itr->path())) {
            if (itr->path().filename().string() == img_dirname) {
                path_found = itr->path();
                return true;
            } else {
                if (find_img_directory(itr->path(), path_found , img_dirname))
                return true;
            }
        }
    }
    return false;
}


enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};

std::vector<std::string> readCSVRow(const std::string &row) {
	//LOG(WARNING) << row << "\n";
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields {""};
    size_t i = 0; // index of the current field
    for (char c : row) {
        switch (state) {
            case CSVState::UnquotedField:
                switch (c) {
                    case ',': // end of field
                              fields.push_back(""); i++;
                              break;
                    case '"': state = CSVState::QuotedField;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedField:
                switch (c) {
                    case '"': state = CSVState::QuotedQuote;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedQuote:
                switch (c) {
                    case ',': // , after closing quote
                              fields.push_back(""); i++;
                              state = CSVState::UnquotedField;
                              break;
                    case '"': // "" -> "
                              fields[i].push_back('"');
                              state = CSVState::QuotedField;
                              break;
                    default:  // end of quote
                              state = CSVState::UnquotedField;
                              break; }
                break;
        }
    }
    return fields;
}

/// Read CSV file, Excel dialect. Accept "quoted fields ""with quotes"""
std::vector<std::vector<std::string>> readCSV(std::istream &in) {
    std::vector<std::vector<std::string>> table;
    std::string row;
    int max_counter = 500;
    int counter = 0;
    while (!in.eof() and counter < max_counter) {
        std::getline(in, row);
        if (in.bad() || in.fail()) {
            break;
        }
        auto fields = readCSVRow(row);
	//for (auto i = fields.begin(); i != fields.end(); ++i)
    	//	std::cout << *i << ' ';
	LOG(WARNING) << fields[2] << "\n";
        table.push_back(fields);
	counter++;
    }
    return table;
}




bool OpenImagesDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    	LOG(INFO) << "Dataset Path: " << datasetPath << '\n';
    	
    	std::string img_filename, img_dirname;
    	path img_dir;
    	path boostDatasetPath(datasetPath);
    	std::string filename = boostDatasetPath.filename().string();
	size_t first = filename.find_first_of('-');
        img_dirname = filename.substr(0, first);


	if (find_img_directory(boostDatasetPath.parent_path().parent_path(), img_dir,  img_dirname)) {
            LOG(INFO) << "Image Directory Found: " << img_dir.string() << '\n';
        } else {
            throw std::invalid_argument("Corresponding Image Directory can't be located, please place it in the same Directory as annotations if you wish to continue without reading images");

        }

	std::ifstream file(datasetPath);
	std::vector<std::vector<std::string>> table = readCSV(file); 
	std::string previousImageID = table[1][0];
	Sample imsample;
	RectRegionsPtr rectRegions(new RectRegions());
	imsample.setSampleID(previousImageID);
	imsample.setColorImage(img_dir.string() + "/" + previousImageID + ".jpg");

	for (int i = 1; i < table.size(); i++) {
		if (previousImageID != table[i][0]) {
			// Create the sample with all the stored bounding boxes and start the list again

			// Create complete new Sample
			imsample.setRectRegions(rectRegions);
			this->map_image_id[previousImageID] = imsample;

			// Restart variables
			rectRegions.reset(new RectRegions());
			imsample.setSampleID(table[i][0]);
			imsample.setColorImage(img_dir.string() + "/" + table[i][0] + ".jpg");
			previousImageID = table[i][0];
		} else {
			// Save the bounding box in a list to then create the Sample
			
			cv::Mat src = cv::imread(img_dir.string() + "/" + previousImageID + ".jpg");
			int imgWidth = src.size().width;
			int imgHeight = src.size().height;
			
			double x, y, w, h;

            		x = atof(table[i][4].c_str()) * imgWidth;
            		y = atof(table[i][6].c_str()) * imgHeight;
            		w = (atof(table[i][5].c_str()) - atof(table[i][4].c_str())) * imgWidth;
            		h = (atof(table[i][7].c_str()) - atof(table[i][6].c_str())) * imgHeight;
			cv::Rect_<double> bounding = cv::Rect_<double>(x , y , w , h);
		       rectRegions->add(bounding, table[i][2],atof(table[i][3].c_str()));
		}
	}

	this->samples.reserve(this->samples.size() + this->map_image_id.size());

    	std::transform (this->map_image_id.begin(), this->map_image_id.end(),back_inserter(this->samples), [] (std::pair<std::string, Sample> const & pair)
	{
		return pair.second;						
	});

	//printDatasetStats();

/*	std::ifstream inFile(datasetPath);
    path boostDatasetPath(datasetPath);
    ClassTypeGeneric typeConverter(this->classNamesFile);
    std::stringstream ss;
LOG(INFO) << "1" << '\n';
    if (inFile) {
  LOG(INFO) << "1.1" << '\n';
	    ss << inFile.rdbuf();
      LOG(INFO) << "1.2" << '\n';
	    inFile.close();
      LOG(INFO) << "1.3" << '\n';
	    LOG(WARNING) << ss.str()<< "\n";
    } else {
      throw std::runtime_error("!! Unable to open json file");
    }
LOG(INFO) << "2" << '\n';*/
}

