#include "xml_wr.h"
namespace nao {
namespace xml {
    /*--------------------------------------------------------------------------------------------------
     --------------------------------------------------------------------------------------------------*/
    Xmlw::Xmlw(const int& type_w, const std::string& file_name)
    {
        _file_name = file_name;
        _type_w = type_w;
        time_t raw_time;
        time(&raw_time);
        switch (_type_w) {
        case 1:
            _fs = cv::FileStorage(_file_name, cv::FileStorage::WRITE);
            _fs << "WIRTE_TIME" << asctime(localtime(&raw_time));

            break;
        case 2:
            _fs = cv::FileStorage(_file_name, cv::FileStorage::APPEND);
            _fs << "WRITE_TIME" << asctime(localtime(&raw_time));
        default:
            break;
        }
        if (!_fs.isOpened()) {
            std::cerr << _file_name << "file open false" << std::endl;
        }
    }

    Xmlw::~Xmlw()
    {
        _fs.release();
        if (_type_w == 1)
            std::cout << "xml write over" << std::endl;
        else
            std::cout << "xml rewrite over" << std::endl;
    }

    cv::FileStorage Xmlw::get_xml_write_object()
    {
        return _fs;
    }

    void Xmlw::writeString(const std::string& key, const std::string& content)
    {
        _fs << key << content;
    }

    void Xmlw::writeString(const std::string& key, const std::vector<std::string>& contents)
    {
        _fs << key << "[" << contents << "]";
    }

    void Xmlw::writeMat(const std::string& key, const cv::Mat& matrix)
    {
        _fs << key << matrix;
    }

    void Xmlw::writeMat(const std::string& key, const std::vector<cv::Mat>& matrixs)
    {
        _fs << key << "[" << matrixs << "]";
    }

    void Xmlw::startParentNode(const std::string& parentNodeName)
    {
        mapsiit it = _parent_node.find(parentNodeName);
        if (it == _parent_node.end()) {
            _fs << parentNodeName << "{";
            _parent_node.insert(std::make_pair(parentNodeName, 1));
        } else
            std::cerr << " " << std::endl;
    }

    void Xmlw::endParentNode()
    {
        _fs << "}";
    }

    /*--------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------*/
    Xmlr::Xmlr(const std::string& file_name)
    {
        _file_name = file_name;
    }

    Xmlr::~Xmlr() { }

    void Xmlr::readString(const std::string& key, std::string& content)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        content = fs[key];
        fs.release();
    }

    void Xmlr::readString(const std::string& key, std::vector<std::string>& contents)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        fs[key][0] >> contents;
        fs.release();
    }

    void Xmlr::readMat(const std::string& key, cv::Mat& matrix)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        fs[key] >> matrix;
        fs.release();
    }

    void Xmlr::readMat(const std::string& key, std::vector<cv::Mat>& matrixs)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        fs[key][0] >> matrixs;
        fs.release();
    }

    /*--------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------*/
    Xml2Dat::Xml2Dat(const std::string& filename)
    {
        _file_name = filename;
        _os = std::ofstream(_file_name, std::ios::out);
    }

    Xml2Dat::~Xml2Dat()
    {
        _os.close();
    };
} // namaspace xml
} // namespace nao
/*----------------------------------------------------------------------------- (C) COPYRIGHT LEI *****END OF FILE------------------------------------------------------------------------------*/