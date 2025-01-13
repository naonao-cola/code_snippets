
target_path="/home/tvt/hjx/"

current_path=$(pwd)

# 判断当前路径是否包含目标路径
if [[ "$current_path" == "$target_path"* ]]; then
    echo "hjx环境已更新"
else
    echo "hjx环境已更新，并来到hjx目录"
    cd "$target_path" || exit
fi

conda activate hjx_base

# alias base='conda activate hjx_base'

alias vhjx='vi ~/hjx/script/hjx_bashrc'
alias vb='vi ~/.bashrc'
alias sb='source ~/.bashrc'
