#include "fs.h"
namespace nao
{
namespace fl
{
/*
非成员函数不应依赖于外部变量, 应尽量置于某个命名空间内.
This are some calsses of file-operation.
参考博客
https://blog.csdn.net/CosmopolitanMe/article/details/80629531
https://blog.csdn.net/sinat_25923849/article/details/78268984
https://www.shuzhiduo.com/A/VGzlVpvy5b/
https://www.cnblogs.com/lkpp/p/OPENFILENAME-lpstrFilter.html
http://gcc.gnu.org/onlinedocs/cpp/_005f_005fhas_005finclude.html
https://www.cctry.com/forum.php?mod=viewthread&tid=47740
https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/scoping/
https://www.cnblogs.com/mathyk/p/11200556.html
https://blog.csdn.net/F_hawk189/article/details/100126602
https://blog.csdn.net/weixin_30855761/article/details/95115196
*/

bool existsDir(const std::string& path)
{
    if (fs::exists(path))
        return true;
    return false;
}

bool isDir(const std::string& path)
{
    if (fs::is_directory(path))
        return true;
    return false;
}

bool creatDir(const std::string& filename)
{
    fs::path dir = filename;
    if (fs::exists(dir))
        return true;
    fs::create_directory(dir);
    if (fs::exists(dir))
        return true;
    else
        return false;
}

bool folderIsEmpty(const std::string& foldername)
{
    fs::path tmp_name = foldername;
    if (fs::is_empty(tmp_name))
        return true;
    return false;
}

std::string base_name(const std::string& filename)
{
    return fs::path(filename).stem().string();
}
std::string parent_parh(const std::string& filename)
{
    return fs::path(filename).parent_path().string();
}

std::string extension_name(const std::string& filename)
{
    return fs::path(filename).extension().string();
}

uintmax_t getFileSize(const std::string& filename)
{
    if (!fs::exists(filename))
        return -1;
    uintmax_t filesize = fs::file_size(filename);
    if (filesize != 0)
        return filesize;
    return 0;
}

uintmax_t removeDir(const std::string& filename)
{
    fs::path dir = filename;
    if (!fs::exists(dir))
        return -1;
    uintmax_t n = fs::remove_all(dir);
    return n;
}

bool getAllFiles(const std::string& path, std::vector<std::string>& files, const std::vector<std::string>& filter_directory)
{
    fs::path file_path = path;
    if (!fs::exists(file_path))
        return false;
    if (fs::is_directory(file_path)) {
        for (auto f : fs::recursive_directory_iterator(file_path)) {
            int foundFlag = 0;   // 过滤文件夹
            for (size_t i = 0; i < filter_directory.size(); i++) {
                std::string::size_type idx;
                fs::path               filter_directory_index = filter_directory[i];
                idx                                           = f.path().string().find(filter_directory_index.string());
                if (idx != std::string::npos) {
                    foundFlag = 1;
                    break;
                }
            }
            if (foundFlag == 0) {
                if (!fs::is_directory(f))
                    files.push_back(f.path().string());
            }
        }
    }
    if (!fs::is_directory(file_path))
        files.push_back(path);
    return true;
}

bool getAllFormatFiles(const std::string& path, std::vector<std::string>& files, const std::string& format, const std::vector<std::string>& filter_directory)
{
    fs::path file_path = path;
    if (!fs::exists(file_path))
        return false;
    std::regex Img(format, std::regex_constants::syntax_option_type::icase);
    if (fs::is_directory(file_path)) {
        for (auto f : fs::recursive_directory_iterator(file_path)) {
            int foundFlag = 0;   // 过滤文件夹
            for (size_t i = 0; i < filter_directory.size(); i++) {
                std::string::size_type idx;
                fs::path               filter_directory_index = filter_directory[i];
                idx                                           = f.path().string().find(filter_directory_index.string());
                if (idx != std::string::npos) {
                    foundFlag = 1;
                    break;
                }
            }
            if (foundFlag == 0) {
                auto fname = f.path().filename().string();
                if (std::regex_match(fname, Img)) {
                    files.push_back(f.path().string());
                }
            }
        }
    }
    else {
        if (std::regex_match(fs::path(path).filename().string(), Img)) {
            files.push_back(path);
        }
    }
    return true;
}

bool getAllFormatFiles(const std::vector<std::string>& path, std::vector<std::string>& files, const std::string& format, const std::vector<std::string>& filter_directory)
{
    for (std::size_t i = 0; i < path.size(); i++) {
        getAllFormatFiles(path[i], files, format, filter_directory);
    }
    return true;
}

int getClassFlag(const std::string& strPath)
{
    std::size_t len  = strPath.size();
    char        drt  = strPath[len - 1];
    int         temp = drt - '0';
    return temp;
}

void getFileList(const std::string& filePath, const int& lable, const std::string& distAll, const std::string& format)
{
    std::vector<std::string> files;
    if (format != "")
        getAllFormatFiles(filePath, files, format);
    if (format == "")
        getAllFiles(filePath, files);
    if (distAll != "")
        std::string filename = distAll;
    std::string   filename = filePath + "\\filelist.txt";
    std::ofstream ofn(filename.c_str(), std::ios::app);
    std::size_t   size = files.size();
    if (lable != 255) {
        for (std::size_t i = 0; i < size; i++) {
            ofn << files[i] << " " << lable << std::endl;
        }
    }
    else {
        for (std::size_t i = 0; i < size; i++) {
            ofn << files[i] << std::endl;
        }
    }
    ofn.close();
}

int removeDuplicateFile(const std::string& orgfile, const std::string& dstfile, std::vector<std::string>& samefilelist, const int flag)
{
    std::vector<std::string> orgfilelist, dstfilelist;
    getAllFiles(orgfile, orgfilelist);
    getAllFiles(dstfile, dstfilelist);
    for (size_t i = 0; i < dstfilelist.size(); i++) {
        std::string dstfilename = fs::path(dstfilelist[i]).filename().string();
        for (std::size_t j = 0; j < orgfilelist.size(); j++) {
            std::string orgfilename = fs::path(orgfilelist[j]).filename().string();
            if (std::strcmp(dstfilename.c_str(), orgfilename.c_str()) == 0) {
                samefilelist.push_back(dstfilelist[i]);
                break;
            }
        }
    }
    if (flag == 1) {
        for (std::size_t i = 0; i < samefilelist.size(); i++)
            removeDir(samefilelist[i]);
    }
    return 0;
}

int alterFileName(const std::string& filename, const std::string& newname)
{
    if (!fs::exists(filename))
        return -2;
    try {
        fs::rename(filename.c_str(), newname.c_str());
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}

bool getCopyFile(const std::string& orgPath, const std::string& dstPath)
{
    if (!fs::exists(orgPath))
        return false;
    fs::path srcpath = orgPath;
    fs::path dstpath = dstPath;
    fs::copy_file(srcpath, dstpath);
    return true;
}

double getDoubleTime()
{
#ifdef WINDOWS
    LARGE_INTEGER t, freq;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&freq);
    return t.QuadPart * 1.0 / freq.QuadPart;
#else
    /* Return the UNIX time in microseconds */
    struct timeval tv;
    long long      ust;
    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust;
#endif
}

void readFileList(const std::string& filename, std::multimap<std::string, int>& lablefilelist)
{
    lablefilelist.clear();
    std::ifstream readData(filename, std::ios::in);
    std::string   buffer;
    int           nClass = 0;
    while (readData) {
        if (getline(readData, buffer)) {
            if (buffer.size() > 0) {
                nClass = getClassFlag(buffer);   // 标签与文件路径之间有一个空格
                std::string temp(buffer, 0, buffer.size() - 2);
                lablefilelist.insert(std::make_pair(temp, nClass));
            }
        }
    }
    readData.close();
}

bool readNoLableFileList(const std::string& filename, std::vector<std::string>& nolablefilelist)
{
    nolablefilelist.clear();
    if (!fs::exists(filename))
        return false;
    std::ifstream readData(filename);
    std::string   buffer;
    while (readData) {
        if (getline(readData, buffer)) {
            nolablefilelist.push_back(buffer);
        }
    }
    readData.close();
    return true;
}

bool filterField(const std::vector<std::string>& srcStringList, std::vector<std::string>& dstStringList, const std::vector<std::string>& format)
{
    if (srcStringList.size() <= 0 || format.size() <= 0)
        return false;
    std::vector<std::regex> expression;
    for (auto f : format) {
        std::regex e(f, std::regex_constants::syntax_option_type::icase);
        expression.push_back(e);
    }
    for (auto s : srcStringList) {
        std::size_t count = 0;
        for (std::size_t i = 0; i < expression.size(); i++) {
            if (std::regex_match(s, expression[i]))
                count++;
        }
        if (count == 0)
            dstStringList.push_back(s);
    }
    return true;
}

bool filterField(const std::vector<std::string>& srcStringList, std::vector<std::string>& dstStringList, const std::string& format)
{
    if (srcStringList.size() <= 0 || format.empty())
        return false;
    std::regex expression(format, std::regex_constants::syntax_option_type::icase);
    for (auto s : srcStringList) {
        if (std::regex_match(s, expression)) {}
        else {
            dstStringList.push_back(s);
        }
    }
    return true;
}

file::~file()
{
    this->close();
}

file::operator bool() const
{
    if (_fout.is_open() || _fin.is_open())
        return true;
    return false;
}

const std::string& file::path() const
{
    if (_path.size() > 0)
        return _path;
    static std::string kPath;
    return kPath;
}

bool file::open(const char* path, std::string mode)
{
    this->close();
    _path = std::string(path);
    if (mode == "r") {
        _fin.open(_path, std::ios::binary);
        return _fin.is_open();
    }
    else if (mode == "r_t") {
        _fin.open(_path, std::ios::in);
        return _fin.is_open();
    }
    else if (mode == "a") {
        _fout.open(_path, std::ios::binary | std::ios::app);
        return _fout.is_open();
    }
    else if (mode == "a_t") {
        _fout.open(_path, std::ios::out | std::ios::app);
        return _fout.is_open();
    }
    else if (mode == "w") {
        _fout.open(_path, std::ios::binary);
        return _fout.is_open();
    }
    else if (mode == "w_t") {
        _fout.open(_path, std::ios::out);
        return _fout.is_open();
    }
    else {
        return false;
    }
}

void file::close()
{
    if (_fin.is_open())
        _fin.close();
    if (_fout.is_open())
        _fout.close();
}

void file::seek(int64_t off, int whence)
{
    static int seekfrom[3] = {std::ios::beg, std::ios::cur, std::ios::end};
    whence                 = seekfrom[whence];
    if (_fin.is_open())
        _fin.seekg(off, whence);
}

size_t file::read(void* s, size_t n)
{
    if (!_fin.is_open())
        return 0;
    const size_t N = 1u << 30;   // 1G
    if (n >= N) {
        throw std::logic_error("单次读取文件过大,超过1G");
    }
    // 判断读取大小是否超限
    {
        std::streampos cur_pos = _fin.tellg();
        _fin.seekg(0, std::ios::end);
        std::streampos remain = _fin.tellg();
        if (n > (size_t)(remain - cur_pos))
            n = remain - cur_pos;
        if (n == 0)
            return 0;
        _fin.seekg(cur_pos);
    }
    _fin.read((char*)s, n);
    return n;
}

std::string file::read(size_t n)
{
    std::string s(n + 1, ' ');
    s.resize(this->read((void*)s.data(), n));
    return s;
}

size_t file::write(const void* s, size_t n)
{
    if (!_fout.is_open())
        return 0;
    const size_t N = 1u << 30;   // 1G
    if (n >= N) {
        throw std::logic_error("单次读取文件过大,超过1G");
    }
    _fout.write((char*)s, n);
    return n;
}
}   // namespace fl
}   // namespace nao