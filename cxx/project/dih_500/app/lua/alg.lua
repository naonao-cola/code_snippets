--V1.0-20241219;
lib = require('portlib')
print(lib.get_versions()) 
--- 函数返回-1表示输入信息解析出错，0表示返回成功
function main(opt,ipt)
    
    channel_id = 0
	channel_name = "--"
	local signal = "--"
	local signal_value = 0.0
    local linetype = 0 -- 校准曲线的序号
	local concentration = "--"
	local concentration_value = 0.0
    local unit = "--"
	--validflag:0成功，  -1试剂卡信息解析出错；-2 参数解析出错, -3 C线失效, -4 校准系数超出合法范围，-5 校准两个通道判定错误，-6 校准卡信息参数不合法
	local validflag = 0 
	local valid_value = 0  -- 信号线失效值
	local output_result = {}
    local result = {}
	local ret = 0
	local coef = 0.0
	local cline_failed = 0
	mode = "--"
	nature_flag = 0      --0 定量  1定性夹心 2 半定量夹心 3 定性竞争 4 半定量竞争

	-- 解析试剂卡信息
	ret,carea_array, area_array, project_id, sample_id, nature_flag = codeinfoandcheck(ipt)
	local nature = "**"
    if(ret < 0)then 
	    print("codeinfoandcheck failed")
	    validflag = ret -- 试剂卡解析出错
	    result[1] = {channel_id = channel_id,channel_name = channel_name,linetype = linetype,signal = signal,concentration_value = concentration_value,concentration = concentration, validflag = validflag, unit = unit, coef=coef, natureflag = nature_flag, nature = nature, mode=mode}
		output_result["output"]  = result;
		optext = cjson.encode(output_result)
		--lib.print_result(opt,optext)
		--return ret
		return ret, optext
	end
	
	-- 判断C线是否有效
	local C_signal = tonumber(carea_array[1])
	local valid_value = tonumber(carea_array[2])
	if(C_signal < valid_value)
	then
		cline_failed = 1 --c线失效
	end

	--计算各个通道的浓度值
    channel_count = #objcardinfo["Chl"]
	local channel_count = #objcardinfo["Chl"] -- 合法性在试剂卡解析中判断
	for i = 1,channel_count,1 do
		channel_name = "--"
		channel_id = 0
		concentration = "--"
		concentration_value = 0.0
		unit = "--"
		validflag = 0
		signal_value = tonumber(area_array[i])
		signal = area_array[i]
		linetype = 0
		nature = "**"
		if(nil ~= channel[i]["CID"])then 
			channel_id = channel[i]["CID"]
		end
		if(nil ~= channel[i]["CNam"])then 
			channel_name = channel[i]["CNam"]
		end
		
		if(nil ~= channel[i]["Unit"])then 
			unit = channel[i]["Unit"]
		end
		local curve = channel[i]["Equa"] 
		decimal = channel[i]["Deci"]  -- 保留小数位数
		curve_count = channel[i]["ECnt"] -- 校准曲线的数目
		bottom_limit = channel[i]["LLmt"]  --最低限制[限制，浓度范围]
		top_limit = channel[i]["HLmt"]  -- 最高限制 
		t_mode = math.floor(tonumber(channel[i]["TMode"]))  -- TMode
		c_mode = math.floor(tonumber(channel[i]["CMode"]))   -- CMode
		mode = table.concat{t_mode, "/", c_mode}
		local gate_array = channel[i]["PGat"] -- 定性门限
		
		if(nil == bottom_limit or nil == top_limit or nil == curve or nil == decimal or nil == curve_count)then 
			validflag = -2 --通道解析出错
			result[1] = {channel_id = channel_id,channel_name = channel_name,linetype = linetype,signal = signal,concentration_value = concentration_value,concentration = concentration, validflag = validflag, unit = unit, coef = coef, natureflag = nature_flag, nature = nature, mode=mode}
			output_result["output"]  = result;
			optext = cjson.encode(output_result)
			--lib.print_result(opt,optext)
			--return -8 --通道参数解析出错
			return -8, optext
		end
		local len_curve = 0 
		len_curve = #curve     
		len_bottom_limit = #bottom_limit 
		len_top_limit = #top_limit 
		
        --没有校准曲线（质控卡和定性的情况）
		if(len_curve == curve_count and 0 == curve_count)then
			validflag = 0
	        if(project_id ~= 105 and 1 == nature_flag or 3 == nature_flag)then--常规样本定性
			    concentration_value = signal_value
				nature = cal_nature(nature_flag, 0, concentration_value, gate_array)
			elseif(project_id ~= 105 and 0 == nature_flag or 2 == nature_flag or 4 == nature_flag)then--常规样本报错
			    concentration_value = signal_value
			    validflag = -6
			end
	    end
		
		--校准或定量或半定量
		if(len_curve == curve_count and  curve_count > 0)then 
            --校准
			validflag = 0
            if (105 == project_id and 1 == sample_id)then 
				if(888 ~= channel_id)then 
					concentration_value = signal_value
					ret, tmp_concentration = get_decimal(concentration_value, decimal) 
					if(0 == ret)then 
						concentration = tmp_concentration
					end

					-- 判定信是否合法，不合法，validflag赋值
					if(nil == tonumber(bottom_limit[1]) or nil == tonumber(top_limit[1]))then 
						validflag = -20 -- 校准参数解析出错 
					elseif(tonumber(bottom_limit[1]) > signal_value or tonumber(top_limit[1]) < signal_value)then 
						validflag = -21 -- 校准信号超出合法范围
					end
				else
					concentration_value = signal_value
					ret, tmp_concentration = get_decimal(concentration_value, decimal)
					if(0 == ret)then 
						concentration = tmp_concentration
					end
				end
			else
				--- 常规测量流程
                if((0 == nature_flag) or (2 == nature_flag) or (4 == nature_flag))then 
					ret,concentration_value,concentration,linetype,flag = cal_concentraiton(signal_value, curve_count, curve, decimal, bottom_limit, top_limit) --计算浓度
					if (0 == ret)then 
						validflag = 0
					elseif(-1 == ret or -3 == ret or -2 == ret)then 
						validflag = -3   --常规测量解析出错
					elseif(-4 == ret)then 
						validflag = -4  -- 常规测量，浓度计算异常  
					end
					---定性
				    nature = cal_nature(nature_flag, flag, concentration_value, gate_array)
				end
			end	
		end
		
        --没有报错，检验C线失效值，如果已报错，C线失效值优先级最低不报
		if(0 == validflag and 1 == cline_failed)then  		
		    validflag = -5 --C线失效值
		end
		result[i] = {}
		result[i] = {channel_id = channel_id,channel_name = channel_name, linetype=linetype,signal=signal,concentration_value=concentration_value,concentration = concentration, validflag = validflag, unit = unit, coef = coef, natureflag = nature_flag, nature = nature, mode=mode}
	end	

	-- 校准合法性判定
	if(105 == project_id and sample_id == 1)then 
		local channel_verify = 0;
		local cal_coef = 0
		for i = 1,channel_count,1 do
			if(result[i].validflag ~= 0)then 
				channel_verify = 1
				break;
			end
		end
		for n = 1,channel_count,1 do
			if(result[n].channel_id == 888)then 
				if(1 == channel_verify)then 
					result[n].validflag = -22 --校准两个通道判定错误
					break;
				else
					local curve = channel[n]["Equa"]
					local param = curve[1]["Para"]--获取曲线参数
					if(nil == curve or nil == param or #param < 3 or tonumber(result[n].signal) == 0 or nil == tonumber(param[1]) or nil == tonumber(param[2]) or nil == tonumber(param[3]))
					then
						---参数有效性确认，有效继续执行，非法跳出
						result[n].validflag = -20  -- 校准参数解析出错
						break
					end
					local len_param = #param
					local low_threshold = tonumber(param[1])
					local high_threshold = tonumber(param[2])
					local target_value = tonumber(param[3])
					cal_coef = target_value / tonumber(result[n].signal)
					if(low_threshold < cal_coef and cal_coef < high_threshold)then 
						result[n].validflag = 0  --有效
					else
						result[n].validflag = -23 --校准系数超出合法范围
					end
				end
			    result[n].coef = cal_coef	
			end
		end
	end
	output_result["output"]  = result;
	optext = cjson.encode(output_result)
	--lib.print_result(opt,optext)
	--return 0
	return 0, optext
end


function cal_nature(nature_flag, flag, concentration, gate_array)
    local nature = "**"
    local i = 0
	local id = 0
	
	if(1 == nature_flag or 2 == nature_flag) --定性半定量夹心
	then
	    nature = "-"
		while(i <= 3)
		do
		    i = i + 1
		    gate = tonumber(gate_array[i])
			if(gate ~= nil and gate ~= 0 and concentration >= gate)then 
			    id = i 
			end
        end	
		if(id == 1)then 
			nature = "±"
		elseif(id == 2)then 
			nature = "+"
		elseif(id == 3)then 
			nature = "++"
		elseif(id == 4)then 
			nature = "+++"
        end	
	elseif(3 == nature_flag or 4 == nature_flag)then --定性半定量竞争
	    local firstflag = 0
	    nature = "-"
	    id = 0
		i = 4
		while(i > 0)
		do
		    gate = tonumber(gate_array[i])
			if(gate ~= nil and gate ~= 0  and firstflag == 0 and concentration <= gate)then
			    firstflag = 1
			    id = i 
			end
			i = i - 1
        end	
		if(id == 1)then 
			nature = "±"
		elseif(id == 2)then 
			nature = "+"
		elseif(id == 3)then 
			nature = "++"
		elseif(id == 4)then 
			nature = "+++"
        end	
	end

	if(1 == flag and 2 == nature_flag)then--半定量夹心小于存在特殊处理，大于不存在特殊处理
		nature = "-"
	end
	
	if(2 == flag and 4 == nature_flag)then--半定量竞争大于存在特殊处理，小于不存在特殊处理
		nature = "-"
	end
	return nature
end


--解析试剂卡信息
function codeinfoandcheck(ipt)
    local root = cjson.decode(ipt)
	local int str_area_cnt = 0
	local area_array = {}
	
    local int str_carea_cnt = 0
	local carea_array = {}

	local sample_id = 0
	local project_id = 0
    local nature_flag = 0
	
	--解析是否为校准
	str_sample = root["sample_id"]
	if(nil == str_sample or nil == tonumber(str_sample) or str_sample == '')then 
	    print("sample_id is nil")
		return -7, carea_array, project_id, sample_id, nature_flag 
	else
	    sample_id = tonumber(str_sample)
	end
	
	-- 解析定性定量参数
	if(nil == root["nature"])then 
	    print("nature_flag is nil")
		return -8, carea_array, area_array, project_id, sample_id, nature_flag
	else
	    nature_flag = tonumber(root["nature"])
	end

	--解析通道信号值
    local str_area = root["signal"]
	if(nil == str_area)then 
		print("str_area is nil")
		return -9, carea_array, area_array, project_id, sample_id, nature_flag
	else
		for s in string.gmatch(str_area, "[^;]+") do
			if(nil ~= s)then 
				str_area_cnt = str_area_cnt + 1
				area_array[str_area_cnt] = s
			end
		end
	end
	
    --分别解析C线信号C线有效值
    local str_cinfo = root["cinfo"]
	if(nil == str_cinfo)then 
		print("str_cinfo is nil")
		return -10, carea_array, area_array, project_id, sample_id, nature_flag
	else
		for s in string.gmatch(str_cinfo, "[^;]+") do
			if(nil ~= s)then 
				str_carea_cnt = str_carea_cnt + 1
				carea_array[str_carea_cnt] = s
			end
		end
	end
	
	
	--解析卡信息
	cardinfo = root["cardinfo"]
	if(nil == cardinfo)then 
	    print("cardinfo is nil")
		return -1, carea_array, area_array, project_id, sample_id, nature_flag
	end
	
	--试剂卡信息解码
	objcardinfo = cjson.decode(cardinfo)
	if(nil == objcardinfo)then 
		print("objcardinfo is nil")
		return -1, carea_array, area_array, project_id, sample_id, nature_flag
	else
	    str_project = objcardinfo["PID"]
		if(nil == str_project)then 
			print("str_project is nil")
			return -11, carea_array, area_array, project_id, sample_id, nature_flag
		else
			project_id = tonumber(str_project)
		end
	
		--解析卡信息中的项目各个项目的信息
		channel = objcardinfo["Chl"]
		print(#objcardinfo["Chl"])
		print(str_area_cnt)
		
		if(nil == channel or nil ==  #objcardinfo["Chl"] or 0 == #objcardinfo["Chl"] or str_area_cnt ~= #objcardinfo["Chl"])then 
			print("channel is nil")
		    return -12, carea_array, area_array, project_id, sample_id, nature_flag
		end
	end
	return 0, carea_array, area_array, project_id, sample_id, nature_flag
end

--判定信号隶属于哪一段; signal信号，线条分段的数目（解析最多支持4段），thresh
function find_curve_id(signal_value, curve, curve_count)
	local index = 0
	local ret = 0
	local threshold
	for i = 1,curve_count,1 do
		threshold = curve[i]["EGat"]--获取曲线阈值
		if(nil == tonumber(threshold[1]) or nil == tonumber(threshold[2]))then 
			ret = -1
			return ret, index
		end
		if(i == 1)then 
			if(signal_value >= tonumber(threshold[1]) and signal_value <= tonumber(threshold[2]))then 
				index = i
			end
		else
			if(signal_value > tonumber(threshold[1]) and signal_value <= tonumber(threshold[2]))then 
				index = i
			end
		end
	end
	if(0 == index)then 
		print("null operation")
	end
	return ret, index
end

-- 返回函数执行的有效性及其浓度
function cal_concentraiton(signal_value, curve_count, curve, decimal, bottom_limit, top_limit)
	local ret = 0;
	local ret1 = 0;
	local curve_id = 0
    local concentration_value = 0.0
	local concentration = "--"
	local tmp_concentration = ""
	local tmp_concentration = 0.0
	if(nil == tonumber(bottom_limit[1]) or nil == tonumber(bottom_limit[2]) or nil == tonumber(top_limit[1]) or nil == tonumber(top_limit[2]))
	then
		ret = -1   --计算浓度时上下限参数无效
		return ret, concentration_value, concentration,curve_id,0
	end

	-- 根据信号与门限寻找对应的线id
	ret1, curve_id = find_curve_id(signal_value, curve, curve_count)
	if(-1 == ret1)then 
		ret = -2  -- 计算浓度时，查找校准曲线id出错
		return ret, concentration_value, concentration, curve_id,0
	end
	if (0 == curve_id)then 
		ret = -2 -- 计算浓度时，查找校准曲线id出错
		return ret,concentration_value,concentration,curve_id,0
	end
	
	--计算浓度
	local linetype = curve[curve_id]["Type"]--获取曲线类型
	local param = curve[curve_id]["Para"]--获取曲线参数
	local int len_param  = #param
	local float a = 0.0
	local float b = 0.0
	local float b = 0.0
	local float d = 0.0
	if(1 == linetype and len_param == 2)then --线性函数
		if(nil == tonumber(param[1]) or nil == tonumber(param[2]))then 
			ret = -3  --校准方程无效
			return ret, concentration_value,concentration,curve_id,0
		end
		a = tonumber(param[1])
		b = tonumber(param[2])
		concentration_value = linear_inverse(a,b,signal_value)
	elseif(3 == linetype and len_param == 4)then --四参数
		if(nil == tonumber(param[1]) or nil == tonumber(param[2]) or nil == tonumber(param[3]) or nil == tonumber(param[4]))then 
			ret = -3  --校准方程无效
			return ret, concentration_value,concentration,curve_id,0
		end
		a = tonumber(param[1])
		b = tonumber(param[2])
		c = tonumber(param[3])
		d = tonumber(param[4])
		concentration_value = four_params_logistic_inverse(a,b,c,d,signal_value)
	elseif(4 == linetype and len_param == 2)then --对数
		if(nil == tonumber(param[1]) or nil == tonumber(param[2]))then 
			ret = -3  --校准方程无效
			return ret, concentration_value,concentration,curve_id,0
		end
		a = tonumber(param[1])
		b = tonumber(param[2])
		concentration_value = log_func_inverse(a,b,signal_value)
	else
		ret = -1  -- 参数无效
	end		
	if(tostring(concentration_value) == "nan" or tostring(concentration_value) == "inf")then 
	   concentration_value = 0.0
	   ret = -4  --反函数无效
	   return ret,concentration_value,concentration,curve_id,0
	end
	
	if(0 == ret)then
		ret, tmp_concentration = get_decimal(concentration_value, decimal)  --保留decimal位小数
		if(0 == ret)then 
			concentration = tmp_concentration
		end
	end
	
   if(tonumber(tmp_concentration) < tonumber(bottom_limit[2]))then 
		curve_id = 11
		concentration = "<"..bottom_limit[2]
		concentration_value = tonumber(bottom_limit[2])
		return ret,concentration_value, concentration,curve_id,1
	else if(tonumber(tmp_concentration) > tonumber(top_limit[2]))then 
		curve_id = 12
		concentration = ">"..top_limit[2]
		concentration_value = tonumber(top_limit[2])
		return ret,concentration_value, concentration, curve_id,2
	end
   return ret,concentration_value,concentration,curve_id,0
end
end

--线性;y=ax+b;a:斜率;b:截距;signal:信号值; 
function linear_inverse(a,b,signal_value)
	return((signal_value-b)/a)
end

--四参数;Y = (a-d）/[1+(X/c)^b]+d；signal:信号值
function four_params_logistic_inverse(a,b,c,d,signal_value)
	return(((a-d)/(signal_value-d)-1)^(1/b)*c)
end

--对数;Y = aln(X）+ b ；signal:信号值
function log_func_inverse(a,b,signal_value)
	return(10^((signal_value-b)/a))
end


-- 保留小数位数，截断
function get_decimal(data, decimal)
	local result = ""
	local str1;
	local str2;
	if type(decimal) ~= "number" then
		return -1, result
	end
	if decimal < 0 then
		return -1, result
	end
	
	if(data < 1e-4)then
	    if(decimal == 0)then
			result = "0"
	    elseif(decimal == 1)then
		    result = "0.0"
	    elseif(decimal == 2)then
		    result = "0.00";
		elseif(decimal == 3)then
		    result = "0.000"
		else
			result = "0.0000"
	    end
		return 0,result
	end
	

	local strdata = tostring(data)
	local dotIndex = string.find(strdata, "%.")
	
	if(nil == dotIndex) then
	    if(decimal == 0)then
	        result = strdata
	    else
	        str1 = "."
	        str2 = string.rep("0",decimal)
		    result = strdata..str1..str2
	    end
	else
		local sublen = string.len(strdata)-dotIndex
		if(decimal == 0)then
			result = string.sub(strdata, 1, dotIndex-1)
		else
			if(sublen > decimal)then
			--剩余位数大于保留位数，截断
				result = string.sub(strdata, 1, dotIndex+decimal)
			elseif(sublen < decimal) then
			--剩余位数小于保留位数，补充
				str1 = string.rep("0",decimal-sublen)
				result = strdata..str1
			else
				result = strdata
			end
		end
	end
	return 0, result
end
