clc;%清除命令窗口的内容
clear;%清除工作空间的所有变量
close all;%关闭所有打开的窗口

file_path =  './stdim/';% 图像文件夹路径 批量处理一个文件夹中的jpg图像  
rlt_excel_dir = 'rlt_excel';%存放结果excel表格
answer_excel = 'answer/answer.xlsx';%标准答案excel表格

img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有jpg格式的图像  
img_num = length(img_path_list);%获取图像总数量 
errno = 0;%错误代码，0表示无误
if img_num > 0 %有图像
    for k = 1:img_num %逐一读取图像  
        image_name = img_path_list(k).name;% 图像名  
        image =  imread(strcat(file_path,image_name));  
        %I{j}=image;
        fprintf('%s\n',strcat(file_path,image_name));% 显示正在处理的图像名 
        
       %% 识别过程
        %预处理
        SHOW=1;
        if numel(image)>2%判断输入图像是彩图还是灰度图
%             gray=rgb2gray(image);%用灰度化函数灰度化
            %自定义灰度化，由于红色的干扰比较大，因此降低红色通道的比重
            R = image(:,:,1);  %通道R
            G = image(:,:,2);  %通道G
            B = image(:,:,3);  %通道B
            gray = (G.*0.45 + B.*0.45 + R.*0.10);%降低R通道权重
        else
            gray=image;
        end
        %滤波与二值化
        g_bw = imbinarize(gray).*1.0;
        Low_High = stretchlim(g_bw, [0.0 0.3]);
        enmed= imadjust(g_bw, Low_High, [ ]);%对比度增强
        med = medfilt2(enmed, [7 5]);%中值滤波
        gausFilter = fspecial('gaussian',[5 5],10);   %高斯滤波器
        blur=imfilter(med,gausFilter,'replicate'); %高斯滤波
        
        bw = imbinarize(blur, max(0.2,graythresh(blur)-0.2));%im2bw
        if SHOW
            figure(),subplot(121),imshow(image);title('原图');
            subplot(122),imshow(gray);title('灰度图');
            figure('name','预处理过程');
            subplot(2, 2, 1);imshow(enmed);title('对比度增强');
            subplot(2, 2, 2);imshow(med);title('中值滤波');
            subplot(2, 2, 3), imshow(blur), title('高斯平滑');
            subplot(2, 2, 4), imshow(bw), title('二值化');
        end
        %% 定位感兴趣区域
        SHOW=1;
        e_in=bw;
        e_in_gray=gray;
        edged_img=edge(e_in,'canny');%边缘检测
        S1 = regionprops(edged_img,'BoundingBox','PixelIdxList');
        max_area = 0;
        for i = 1:length(S1)%寻找最大BoundingBox，BoundingBox格式[xmin,ymin,width,hight]
            area = S1(i).BoundingBox(3)*S1(i).BoundingBox(4); 
            if area>max_area
               max_area = area;
               pos = i;
            end
        end
        bbox = round(S1(pos).BoundingBox);
        bbox(3:4)=bbox(3:4) + 1;
        ROIofbw=e_in(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI表示感兴趣区域
        ROI_ORI=e_in_gray(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%####ORI 
        e_in_new=im2double(zeros(size(e_in)));
        e_in_new(S1(pos).PixelIdxList)=1;
        CNT_ROI=e_in_new(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI表示感兴趣区域
        if SHOW
            figure('name','边缘检测'),imshow(edged_img);
            figure('name','ROI');
            subplot(1,2,1);
            imshow(ROIofbw);title('感兴趣区域二值图');
            subplot(1,2,2);imshow(CNT_ROI);title('感兴趣区域轮廓');
            imwrite(CNT_ROI,'./refer/CNT_ROI.jpg');
        end
        %% 倾斜校正 使用hough直线检测
        SHOW = 1;%控制是否显示中间过程
        %输入参数
        ROIofBW = ROIofbw;
%         CNT_ROI=CNT_ROI;
        [H,Theta,Rho] = hough(CNT_ROI);
        [hight,width]=size(CNT_ROI);
        P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));%找到信号最强的几条直线
        lines = houghlines(CNT_ROI,Theta,Rho,P,'MinLength',hight/80);%'FillGap',3,
        max_len=0;%找最长竖向直线.
        pos = 1;
        for i = 1:length(lines)
            p1 = lines(i).point1;
            p2 = lines(i).point2;
            len = norm(p1-p2);%sqrt((p1(1)-p2(1))^2+(p1(2)-p2(2))^2);%范数表达 线段长度  
            if len>max_len && p1(1)>width/2 && p2(1)>width/2 &&abs(lines(i).theta)<40%角度绝对值
                max_len =len;
                pos = i;
            end
        end
        if length(lines)==1
            rotate_angle=lines.theta;%求旋转角
        else
            rotate_angle=lines(pos).theta;%求旋转角
        end
        rotated_ROIofBW = imrotate(ROIofBW,rotate_angle);
        rotated_ROI_ORI = imrotate(ROI_ORI,rotate_angle);%####ORI
        rotated_CNT_ROI = imrotate(CNT_ROI,rotate_angle);
        if SHOW
            figure,subplot(121),imshow(ROIofBW);title('倾斜校正前');
            hold on
            xy = [lines(pos).point1; lines(pos).point2];
            plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
            hold off;
            subplot(122),imshow(rotated_ROIofBW);title('倾斜校正后');
            imwrite(rotated_ROIofBW,'refer/rotated_ROIofBW.jpg')
        end
        %倾斜校正后得到的图像是rotated_ROIofBW和rotated_CNT_ROI
        %% 再次定位感兴趣区域
        SHOW=1;
        e_in=rotated_ROIofBW;
        e_in_gray = rotated_ROI_ORI;%####ORI
        edged_img=edge(e_in,'canny');%边缘检测
        S1 = regionprops(edged_img,'BoundingBox','PixelIdxList');
        max_area = 0;
        for i = 1:length(S1)%寻找最大BoundingBox
            area = S1(i).BoundingBox(3)*S1(i).BoundingBox(4); 
            if area>max_area
               max_area = area;
               pos = i;
            end
        end
        bbox = round(S1(pos).BoundingBox);
        bbox(3:4)=bbox(3:4) + 1;
        ROIofbw=e_in(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%感兴趣区域二值图
        ROI_ORI=e_in_gray(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI表示感兴趣区域，ORI原图####ORI   感兴趣区域灰度图
        e_in_new=im2double(zeros(size(e_in)));
        e_in_new(S1(pos).PixelIdxList)=1;
        CNT_ROI=e_in_new(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI表示感兴趣区域
        if SHOW
            figure('name','RE_ROI');
            subplot(1,2,1);
            imshow(ROIofbw);title('再次定位感兴趣区域二值图');
            subplot(1,2,2);imshow(CNT_ROI);title('再次定位感兴趣区域轮廓');
%             imwrite(CNT_ROI,'./refer/CNT_ROI.jpg');%有必要的话可以保存下来
        end
        
       
        %%% 畸变校正
        SHOW=1;%控制是否显示中间过程

        rotated_CNT_ROI = CNT_ROI;
        rotated_ROIofBW = ROIofbw;
        %% harris寻找角点
        se = strel('square',5);%结构元素（用于腐蚀膨胀或开闭运算）
        dilated_img=imdilate(rotated_CNT_ROI,se);%膨胀
        se = strel('square',5);%结构元素（用于腐蚀膨胀或开闭运算）
        closed_img=imerode(dilated_img,se);%腐蚀
        corners = detectHarrisFeatures(closed_img);
        if SHOW
            figure('name','腐蚀膨胀');
            subplot(121);imshow(rotated_CNT_ROI,'InitialMagnification','fit');title('膨胀腐蚀(闭运算)前');
            subplot(122);imshow(closed_img,'InitialMagnification','fit');title('膨胀腐蚀(闭运算)后');

            figure('name','角点');
            imshow(closed_img,'InitialMagnification','fit'),title('角点检测'); hold on;
            plot(corners);hold off
        end
        %% 四角精确定位--以便作投影变换
        SHOW=1;
        
        conrs=corners;
        %conrs=corners.selectStrongest(300);%可以只在信号最强的有限个点以内找角点，减少计算量
        %Inf“正无穷”
        min_dis_ul = Inf; %ul=upper left corner  记录左上角到最近角点距离   Inf是无穷大
        min_dis_ur = Inf; %ur=upper right corner 记录右上角到最近角点距离
        min_dis_br = Inf; %lr=bottom right corner记录右下角到最近角点距离
        min_dis_bl = Inf; %bl=bottom left corner 记录左下角到最近角点距离

        nearby_ul_xy = [0 0]; %ul=upper left corner  记录左上角最近角点坐标
        nearby_ur_xy = [0 0]; %ur=upper right corner 记录右上角最近角点坐标
        nearby_br_xy = [0 0]; %lr=bottom right corner记录右下角最近角点坐标
        nearby_bl_xy = [0 0]; %bl=bottom left corner 记录左下角最近角点坐标
        [h,w]=size(rotated_CNT_ROI);
        
        ul_c=[0 0];          %upper left corner   左上角坐标
        ur_c=[w 0];      %upper right corner  右上角坐标
        br_c=[w h]; %bottom right corner 右下角坐标
        bl_c=[0 h];     %bottom left corner  左下角坐标
        for i=1:length(conrs)
            dis = norm(conrs(i).Location-ul_c);%左上 %%Location格式为[x,y]
            if dis<min_dis_ul
                min_dis_ul = dis;
                index_ul=i;
            end
            
            dis = norm(conrs(i).Location-ur_c);%右上
            if dis<min_dis_ur
                min_dis_ur = dis;
                index_ur=i;
            end

            dis = norm(conrs(i).Location-bl_c);%左下
            if dis<min_dis_bl
                min_dis_bl = dis;
                index_bl=i;
            end

            dis = norm(conrs(i).Location-br_c);%右下
            if dis<min_dis_br
                min_dis_br = dis;
                index_br=i;
            end
        end
        p_ul = round(conrs(index_ul).Location);%p表示point，ul表示upper left，左上角角点
        p_ur = round(conrs(index_ur).Location);%右上角角点
        p_bl = round(conrs(index_bl).Location);%左下角角点
        p_br = round(conrs(index_br).Location);%br表示bottom right，右下角角点
        %显示
        if SHOW
            figure('name','精确定位4个角点');
            imshow(closed_img,'InitialMagnification','fit');title('精确定位4个角点');hold on;
            
            plot(p_br(1),p_br(2),'x','LineWidth', 4, 'Color', 'red');
            plot(p_ur(1),p_ur(2),'x','LineWidth', 4, 'Color', 'blue');
            plot(p_bl(1),p_bl(2),'x','LineWidth', 4, 'Color', 'yellow');
            plot(p_ul(1),p_ul(2),'x','LineWidth', 4, 'Color', 'red');
            hold off
        end
        
        %确定p_ul对应的是左上角哪一个角点
        %用两个点确定一条直线，一是ur（右上角点）和bl（左下角点）的中点，二是br（右下角点）,如果ul在直线上方，就是‘L’类型，否之为'R'
        mid_ur_bl = zeros(1,2);
        mid_ur_bl(1,1)=round((p_ur(1,1)+p_bl(1,1))/2);
        mid_ur_bl(1,2)=round((p_ur(1,2)+p_bl(1,2))/2);
        x1=mid_ur_bl(1,1);
        y1=mid_ur_bl(1,2);
        x2=p_br(1,1);
        y2=p_br(1,2);
        Y=(y2-y1)*(p_ul(1,1)-x1)/(x2-x1)+y1;%(X-x1)(y2-y1)=(Y-y1)(x2-x1)-->Y=(y2-y1)(X-x1)/(x2-x1)+y1
        if Y>=0
            ul_type = 'L';
        else
            ul_type = 'R';
        end
        %返回 p_ul，p_ur，p_bl，p_br，ul_type
        
        %% 投影变换，解决透视变形问题
        SHOW = 1;
        %原来的点
        x = [p_ul(1) p_ur(1) p_br(1) p_bl(1)];
        y = [p_ul(2) p_ur(2) p_br(2) p_bl(2)];     
        %校正后的点
        [height,width] = size(rotated_ROIofBW); 
        if ul_type=='L'
            ul_x=1;ul_y=round(150.0/3530*height);
        else
            ul_x=(140.0/2472*width);ul_y=1;
        end
        X=[ ul_x width  width    1    ];
        Y=[ ul_y   1    height height ];
        tform = fitgeotrans([ x' y'],[ X' Y'],'Projective');
        outputView = imref2d([height,width]);

        corrected_im_gray = imwarp(rotated_ROI_ORI,tform, 'OutputView', outputView);%这个图只是为了展示用，实际识别没有用到
        corrected_img = ~imwarp(~rotated_ROIofBW,tform, 'OutputView', outputView);%这里的两次取反操作细节是反复实验的结果，
                                                                                   %仿射变换会使总体的灰度值降低，降低图像质量，
                                                                                   %先取反，再做仿射变换，再取反反而会增强原信号                                                                           %
        %显示
        if SHOW
            figure,subplot(121),imshow(rotated_ROIofBW,'InitialMagnification','fit');title('畸变校正(投影变换)前');
            subplot(122),imshow(corrected_img,'InitialMagnification','fit');title('畸变校正(投影变换)后');
        end
        
        %% 填涂区域划分
        SHOW =1;
        bw_im = ~corrected_img;
        [h,w]=size(bw_im);
        %寻找锚点/定位点
        x_start = w-int32(380/3774*w);
        anchor_region=bw_im(1:h,x_start:w-10);
        [rh,rw]=size(anchor_region);
        anchor_region = bwareaopen(anchor_region,floor(rh/200*rw/12));%去掉小的连通域
%         figure,imshow(anchor_region);
        STATS = regionprops(anchor_region,'basic');%basic:Area,Centroid,BoundingBox
        centro_y = [];%锚点中心点
        for i = 1:length(STATS)
            centro_y = [centro_y round(STATS(i).Centroid(2))];
        end
        [val,i_max]=max(centro_y);
        [val,i_min]=min(centro_y);
        start_y = STATS(i_min).BoundingBox(2)-STATS(i_min).BoundingBox(4)*0.5;
        end_y = STATS(i_max).BoundingBox(2) + STATS(i_max).BoundingBox(4)*1.5;
        %划分格子
        se = strel('square',3);%结构元素（用于腐蚀膨胀或开闭运算）
        e_bw_im=imerode(bw_im,se);%腐蚀
        e_bw_im = bwareaopen(e_bw_im,100);
        se = strel('square',11);%结构元素（用于腐蚀膨胀或开闭运算）
        e_bw_im=imdilate(e_bw_im,se);%膨胀

        dy = (end_y-start_y)/length(STATS);
        %准考证号
        kh_start_x = int32(935.0/2530*w);
        kh_end_x = int32(2225.0/2530*w);
        kh_start_y = int32(start_y+dy*5);
        kh_end_y = int32(start_y+dy*15);
        %kh_area = e_bw_im(kh_start_y:kh_end_y,kh_start_x:kh_end_x);
        dx = (kh_end_x-kh_start_x)*1.0/9;
        kh_dy = (kh_end_y-kh_start_y)*0.1;
        %figure,imshow(kh_area),title('kh_area');
        %% 识别考号
        thresh_optarea = dx*kh_dy/5;
        kh_num=[];
%         figure('name','boxes');
        for i=1:9
            cnt = 0;%判断考号一列填了几个选项
            for j=1:10
                %box = kh_area(int32(kh_dy*(j-1)+1):int32(kh_dy*(j)),int32(kh_dx*(i-1)+1):int32(kh_dx*(i)-1));
                b_x = int32(kh_start_x + dx*(i-1)+1);
                b_y = int32(kh_start_y + kh_dy*(j-1)+1);
                box = e_bw_im(b_y:int32(b_y+kh_dy),b_x:int32(b_x+dx));
%                 subplot(9,10,9*(i-1)+j),imshow(box);
                opt = get_opt(box,thresh_optarea);
                if opt
                    cnt=cnt+1;
                    if cnt>1
                        errno = 2;%考号某列多填了
                        disp('考号某列填涂错误，多填');
                        return;
                    end
                    kh_num = [kh_num j-1];
                end
            end
            if cnt==0
                errno = 3;%考号某列漏填了
                disp('考号某列填涂错误，漏填');
                return;
            end
        end
        
        
        
        if SHOW
            merge = uint8(double(e_bw_im).*127.0+double(corrected_im_gray)*0.5);
            figure('name','填涂区域划分'),imshow(e_bw_im),title('填涂区域划分'),hold on;%e_bw_im 或者 merge
            %画横线
            for i = 1:length(STATS)
                plot(x_start+STATS(i).Centroid(1),STATS(i).Centroid(2),'x','LineWidth',4,'Color','b');
                xy = [2 start_y+dy*(i-1); w-2 start_y+dy*(i-1)];
                plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            end
            xy = [2 start_y+dy*(i); w-2 start_y+dy*(i)];
            plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            %画考号区域竖线
            for i = 1:10
                xy = [kh_start_x+(i-1)*dx start_y; kh_start_x+(i-1)*dx kh_end_y];
                plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            end
%             xy = [kh_start_x+(i)*kh_dx start_y; kh_start_x+(i)*kh_dx kh_end_y];
%             plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','b');
            
        end
        %% 识别选项
        opt_start_x =kh_start_x - 5*dx;
        opt_end_x = kh_start_x - dx;
        opt_start_y = round(kh_end_y - kh_dy*2.0);
        opt_end_y = end_y;
        opt_dx = dx;
        if SHOW
            for i = 0 : 14
                xy = [opt_start_x+(i)*opt_dx opt_start_y; opt_start_x+(i)*opt_dx opt_end_y];
                plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            end
            hold off;
        end
        
        opt_rlt=zeros(1,40);
        opt_map=cell(1,40);
        %1-5
        for i=1:5
            opt_map{i}=e_bw_im(opt_start_y+(i-1)*dy:opt_start_y+(i)*dy,opt_start_x:opt_start_x+4*dx);
        end
        % 6-10
%         figurre();
        for i=7:11
            opt_map{i-1}=e_bw_im(opt_start_y+(i-1)*dy:opt_start_y+(i)*dy,opt_start_x:opt_start_x+4*dx);
%             subplot(5,1,i-6),imshow(opt_map{i-1});
        end
        % 11-15
%         figure();
        for i=13:17
            opt_map{i-2}=e_bw_im(opt_start_y+(i-1)*dy:opt_start_y+(i)*dy,opt_start_x:opt_start_x+4*dx);
%             subplot(5,1,i-12),imshow(opt_map{i-2});
        end
        
        % 16-20
%         figure();
        for i=19:23
            opt_map{i-3}=e_bw_im(opt_start_y+(i-1)*dy:opt_start_y+(i)*dy,opt_start_x:opt_start_x+4*dx);
%             subplot(5,1,i-18),imshow(opt_map{i-3});
        end
        % 21-25
%         figure();
        for i=21:25
            opt_map{i}=e_bw_im(opt_start_y+(i-15)*dy:opt_start_y+(i-14)*dy,opt_start_x+5*dx:opt_start_x+9*dx);
%             subplot(5,1,i-20),imshow(opt_map{i});
        end
        % 23-30
%         figure();
        for i=26:30
            opt_map{i}=e_bw_im(opt_start_y+(i-14)*dy:opt_start_y+(i-13)*dy,opt_start_x+5*dx:opt_start_x+9*dx);
%             subplot(5,1,i-25),imshow(opt_map{i});
        end
        %选项
        % 31-35
%         figure();
        for i=31:35
            opt_map{i}=e_bw_im(opt_start_y+(i-13)*dy:opt_start_y+(i-12)*dy,opt_start_x+5*dx:opt_start_x+9*dx);
%             subplot(5,1,i-30),imshow(opt_map{i});
        end
        % 36-40
%         figure();
        for i=36:40
            opt_map{i}=e_bw_im(opt_start_y+(i-30)*dy:opt_start_y+(i-29)*dy,opt_start_x+10*dx:opt_start_x+14*dx);
%             subplot(5,1,i-35),imshow(opt_map{i});
        end
        %% 选项识别
        ANS = {'A','B','C','D','wrong'};
        for i=1:40
            opt_rlt(i) = get_rlt(opt_map{i});
        end
        %% 数据处理
        %输入opt_rlt，正确答案的Excel 输出
        filename = strcat(rlt_excel_dir,'/',char(kh_num+'0'),'.xlsx');%考号为文件名
        excel_gen = cell(42,4);
        excel_gen{1,1}='题号';
        excel_gen{1,2}='本考生答案';
        excel_gen{1,3}='正确答案';
        excel_gen{1,4}='得分';
        if  exist(answer_excel) ~= 0
            [num,txt,raw]=xlsread(answer_excel);
        else
            errno = 2;
            disp('没有找到标准答案Excel表格');
            return ;
        end
        score = 0;
        count = 0;%正确个数
        len = min(length(raw)-1,length(opt_rlt));
        for i=1:len
            excel_gen{i+1,1}=i;
            excel_gen{i+1,2}=ANS{opt_rlt(i)};
            excel_gen{i+1,3}=raw{i+1,2};
            if ANS{opt_rlt(i)}==raw{i+1,2}
                score = score + raw{i+1,3};
                excel_gen{i+1,4}=raw{i+1,3};
                count = count + 1;
            else
                excel_gen{i+1,4}=0;
            end
        end
%         excel_gen{len+2,1}='考号';
        excel_gen{len+2,1}=strcat('考号: ',char(kh_num+'0'));
        excel_gen{len+2,3}='总分';
        excel_gen{len+2,4}=score;
        %将结果输出到Excel
        if  exist(rlt_excel_dir)==0
            mkdir(rlt_excel_dir);
        end
        
        xlswrite(filename,excel_gen);
        %uiwait;
        
    end  
end