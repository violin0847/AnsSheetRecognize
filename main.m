clc;%�������ڵ�����
clear;%��������ռ�����б���
close all;%�ر����д򿪵Ĵ���

file_path =  './stdim/';% ͼ���ļ���·�� ��������һ���ļ����е�jpgͼ��  
rlt_excel_dir = 'rlt_excel';%��Ž��excel���
answer_excel = 'answer/answer.xlsx';%��׼��excel���

img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��  
img_num = length(img_path_list);%��ȡͼ�������� 
errno = 0;%������룬0��ʾ����
if img_num > 0 %��ͼ��
    for k = 1:img_num %��һ��ȡͼ��  
        image_name = img_path_list(k).name;% ͼ����  
        image =  imread(strcat(file_path,image_name));  
        %I{j}=image;
        fprintf('%s\n',strcat(file_path,image_name));% ��ʾ���ڴ����ͼ���� 
        
       %% ʶ�����
        %Ԥ����
        SHOW=1;
        if numel(image)>2%�ж�����ͼ���ǲ�ͼ���ǻҶ�ͼ
%             gray=rgb2gray(image);%�ûҶȻ������ҶȻ�
            %�Զ���ҶȻ������ں�ɫ�ĸ��űȽϴ���˽��ͺ�ɫͨ���ı���
            R = image(:,:,1);  %ͨ��R
            G = image(:,:,2);  %ͨ��G
            B = image(:,:,3);  %ͨ��B
            gray = (G.*0.45 + B.*0.45 + R.*0.10);%����Rͨ��Ȩ��
        else
            gray=image;
        end
        %�˲����ֵ��
        g_bw = imbinarize(gray).*1.0;
        Low_High = stretchlim(g_bw, [0.0 0.3]);
        enmed= imadjust(g_bw, Low_High, [ ]);%�Աȶ���ǿ
        med = medfilt2(enmed, [7 5]);%��ֵ�˲�
        gausFilter = fspecial('gaussian',[5 5],10);   %��˹�˲���
        blur=imfilter(med,gausFilter,'replicate'); %��˹�˲�
        
        bw = imbinarize(blur, max(0.2,graythresh(blur)-0.2));%im2bw
        if SHOW
            figure(),subplot(121),imshow(image);title('ԭͼ');
            subplot(122),imshow(gray);title('�Ҷ�ͼ');
            figure('name','Ԥ�������');
            subplot(2, 2, 1);imshow(enmed);title('�Աȶ���ǿ');
            subplot(2, 2, 2);imshow(med);title('��ֵ�˲�');
            subplot(2, 2, 3), imshow(blur), title('��˹ƽ��');
            subplot(2, 2, 4), imshow(bw), title('��ֵ��');
        end
        %% ��λ����Ȥ����
        SHOW=1;
        e_in=bw;
        e_in_gray=gray;
        edged_img=edge(e_in,'canny');%��Ե���
        S1 = regionprops(edged_img,'BoundingBox','PixelIdxList');
        max_area = 0;
        for i = 1:length(S1)%Ѱ�����BoundingBox��BoundingBox��ʽ[xmin,ymin,width,hight]
            area = S1(i).BoundingBox(3)*S1(i).BoundingBox(4); 
            if area>max_area
               max_area = area;
               pos = i;
            end
        end
        bbox = round(S1(pos).BoundingBox);
        bbox(3:4)=bbox(3:4) + 1;
        ROIofbw=e_in(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI��ʾ����Ȥ����
        ROI_ORI=e_in_gray(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%####ORI 
        e_in_new=im2double(zeros(size(e_in)));
        e_in_new(S1(pos).PixelIdxList)=1;
        CNT_ROI=e_in_new(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI��ʾ����Ȥ����
        if SHOW
            figure('name','��Ե���'),imshow(edged_img);
            figure('name','ROI');
            subplot(1,2,1);
            imshow(ROIofbw);title('����Ȥ�����ֵͼ');
            subplot(1,2,2);imshow(CNT_ROI);title('����Ȥ��������');
            imwrite(CNT_ROI,'./refer/CNT_ROI.jpg');
        end
        %% ��бУ�� ʹ��houghֱ�߼��
        SHOW = 1;%�����Ƿ���ʾ�м����
        %�������
        ROIofBW = ROIofbw;
%         CNT_ROI=CNT_ROI;
        [H,Theta,Rho] = hough(CNT_ROI);
        [hight,width]=size(CNT_ROI);
        P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));%�ҵ��ź���ǿ�ļ���ֱ��
        lines = houghlines(CNT_ROI,Theta,Rho,P,'MinLength',hight/80);%'FillGap',3,
        max_len=0;%�������ֱ��.
        pos = 1;
        for i = 1:length(lines)
            p1 = lines(i).point1;
            p2 = lines(i).point2;
            len = norm(p1-p2);%sqrt((p1(1)-p2(1))^2+(p1(2)-p2(2))^2);%������� �߶γ���  
            if len>max_len && p1(1)>width/2 && p2(1)>width/2 &&abs(lines(i).theta)<40%�ǶȾ���ֵ
                max_len =len;
                pos = i;
            end
        end
        if length(lines)==1
            rotate_angle=lines.theta;%����ת��
        else
            rotate_angle=lines(pos).theta;%����ת��
        end
        rotated_ROIofBW = imrotate(ROIofBW,rotate_angle);
        rotated_ROI_ORI = imrotate(ROI_ORI,rotate_angle);%####ORI
        rotated_CNT_ROI = imrotate(CNT_ROI,rotate_angle);
        if SHOW
            figure,subplot(121),imshow(ROIofBW);title('��бУ��ǰ');
            hold on
            xy = [lines(pos).point1; lines(pos).point2];
            plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
            hold off;
            subplot(122),imshow(rotated_ROIofBW);title('��бУ����');
            imwrite(rotated_ROIofBW,'refer/rotated_ROIofBW.jpg')
        end
        %��бУ����õ���ͼ����rotated_ROIofBW��rotated_CNT_ROI
        %% �ٴζ�λ����Ȥ����
        SHOW=1;
        e_in=rotated_ROIofBW;
        e_in_gray = rotated_ROI_ORI;%####ORI
        edged_img=edge(e_in,'canny');%��Ե���
        S1 = regionprops(edged_img,'BoundingBox','PixelIdxList');
        max_area = 0;
        for i = 1:length(S1)%Ѱ�����BoundingBox
            area = S1(i).BoundingBox(3)*S1(i).BoundingBox(4); 
            if area>max_area
               max_area = area;
               pos = i;
            end
        end
        bbox = round(S1(pos).BoundingBox);
        bbox(3:4)=bbox(3:4) + 1;
        ROIofbw=e_in(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%����Ȥ�����ֵͼ
        ROI_ORI=e_in_gray(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI��ʾ����Ȥ����ORIԭͼ####ORI   ����Ȥ����Ҷ�ͼ
        e_in_new=im2double(zeros(size(e_in)));
        e_in_new(S1(pos).PixelIdxList)=1;
        CNT_ROI=e_in_new(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1);%ROI��ʾ����Ȥ����
        if SHOW
            figure('name','RE_ROI');
            subplot(1,2,1);
            imshow(ROIofbw);title('�ٴζ�λ����Ȥ�����ֵͼ');
            subplot(1,2,2);imshow(CNT_ROI);title('�ٴζ�λ����Ȥ��������');
%             imwrite(CNT_ROI,'./refer/CNT_ROI.jpg');%�б�Ҫ�Ļ����Ա�������
        end
        
       
        %%% ����У��
        SHOW=1;%�����Ƿ���ʾ�м����

        rotated_CNT_ROI = CNT_ROI;
        rotated_ROIofBW = ROIofbw;
        %% harrisѰ�ҽǵ�
        se = strel('square',5);%�ṹԪ�أ����ڸ�ʴ���ͻ򿪱����㣩
        dilated_img=imdilate(rotated_CNT_ROI,se);%����
        se = strel('square',5);%�ṹԪ�أ����ڸ�ʴ���ͻ򿪱����㣩
        closed_img=imerode(dilated_img,se);%��ʴ
        corners = detectHarrisFeatures(closed_img);
        if SHOW
            figure('name','��ʴ����');
            subplot(121);imshow(rotated_CNT_ROI,'InitialMagnification','fit');title('���͸�ʴ(������)ǰ');
            subplot(122);imshow(closed_img,'InitialMagnification','fit');title('���͸�ʴ(������)��');

            figure('name','�ǵ�');
            imshow(closed_img,'InitialMagnification','fit'),title('�ǵ���'); hold on;
            plot(corners);hold off
        end
        %% �ĽǾ�ȷ��λ--�Ա���ͶӰ�任
        SHOW=1;
        
        conrs=corners;
        %conrs=corners.selectStrongest(300);%����ֻ���ź���ǿ�����޸��������ҽǵ㣬���ټ�����
        %Inf�������
        min_dis_ul = Inf; %ul=upper left corner  ��¼���Ͻǵ�����ǵ����   Inf�������
        min_dis_ur = Inf; %ur=upper right corner ��¼���Ͻǵ�����ǵ����
        min_dis_br = Inf; %lr=bottom right corner��¼���½ǵ�����ǵ����
        min_dis_bl = Inf; %bl=bottom left corner ��¼���½ǵ�����ǵ����

        nearby_ul_xy = [0 0]; %ul=upper left corner  ��¼���Ͻ�����ǵ�����
        nearby_ur_xy = [0 0]; %ur=upper right corner ��¼���Ͻ�����ǵ�����
        nearby_br_xy = [0 0]; %lr=bottom right corner��¼���½�����ǵ�����
        nearby_bl_xy = [0 0]; %bl=bottom left corner ��¼���½�����ǵ�����
        [h,w]=size(rotated_CNT_ROI);
        
        ul_c=[0 0];          %upper left corner   ���Ͻ�����
        ur_c=[w 0];      %upper right corner  ���Ͻ�����
        br_c=[w h]; %bottom right corner ���½�����
        bl_c=[0 h];     %bottom left corner  ���½�����
        for i=1:length(conrs)
            dis = norm(conrs(i).Location-ul_c);%���� %%Location��ʽΪ[x,y]
            if dis<min_dis_ul
                min_dis_ul = dis;
                index_ul=i;
            end
            
            dis = norm(conrs(i).Location-ur_c);%����
            if dis<min_dis_ur
                min_dis_ur = dis;
                index_ur=i;
            end

            dis = norm(conrs(i).Location-bl_c);%����
            if dis<min_dis_bl
                min_dis_bl = dis;
                index_bl=i;
            end

            dis = norm(conrs(i).Location-br_c);%����
            if dis<min_dis_br
                min_dis_br = dis;
                index_br=i;
            end
        end
        p_ul = round(conrs(index_ul).Location);%p��ʾpoint��ul��ʾupper left�����Ͻǽǵ�
        p_ur = round(conrs(index_ur).Location);%���Ͻǽǵ�
        p_bl = round(conrs(index_bl).Location);%���½ǽǵ�
        p_br = round(conrs(index_br).Location);%br��ʾbottom right�����½ǽǵ�
        %��ʾ
        if SHOW
            figure('name','��ȷ��λ4���ǵ�');
            imshow(closed_img,'InitialMagnification','fit');title('��ȷ��λ4���ǵ�');hold on;
            
            plot(p_br(1),p_br(2),'x','LineWidth', 4, 'Color', 'red');
            plot(p_ur(1),p_ur(2),'x','LineWidth', 4, 'Color', 'blue');
            plot(p_bl(1),p_bl(2),'x','LineWidth', 4, 'Color', 'yellow');
            plot(p_ul(1),p_ul(2),'x','LineWidth', 4, 'Color', 'red');
            hold off
        end
        
        %ȷ��p_ul��Ӧ�������Ͻ���һ���ǵ�
        %��������ȷ��һ��ֱ�ߣ�һ��ur�����Ͻǵ㣩��bl�����½ǵ㣩���е㣬����br�����½ǵ㣩,���ul��ֱ���Ϸ������ǡ�L�����ͣ���֮Ϊ'R'
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
        %���� p_ul��p_ur��p_bl��p_br��ul_type
        
        %% ͶӰ�任�����͸�ӱ�������
        SHOW = 1;
        %ԭ���ĵ�
        x = [p_ul(1) p_ur(1) p_br(1) p_bl(1)];
        y = [p_ul(2) p_ur(2) p_br(2) p_bl(2)];     
        %У����ĵ�
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

        corrected_im_gray = imwarp(rotated_ROI_ORI,tform, 'OutputView', outputView);%���ͼֻ��Ϊ��չʾ�ã�ʵ��ʶ��û���õ�
        corrected_img = ~imwarp(~rotated_ROIofBW,tform, 'OutputView', outputView);%���������ȡ������ϸ���Ƿ���ʵ��Ľ����
                                                                                   %����任��ʹ����ĻҶ�ֵ���ͣ�����ͼ��������
                                                                                   %��ȡ������������任����ȡ����������ǿԭ�ź�                                                                           %
        %��ʾ
        if SHOW
            figure,subplot(121),imshow(rotated_ROIofBW,'InitialMagnification','fit');title('����У��(ͶӰ�任)ǰ');
            subplot(122),imshow(corrected_img,'InitialMagnification','fit');title('����У��(ͶӰ�任)��');
        end
        
        %% ��Ϳ���򻮷�
        SHOW =1;
        bw_im = ~corrected_img;
        [h,w]=size(bw_im);
        %Ѱ��ê��/��λ��
        x_start = w-int32(380/3774*w);
        anchor_region=bw_im(1:h,x_start:w-10);
        [rh,rw]=size(anchor_region);
        anchor_region = bwareaopen(anchor_region,floor(rh/200*rw/12));%ȥ��С����ͨ��
%         figure,imshow(anchor_region);
        STATS = regionprops(anchor_region,'basic');%basic:Area,Centroid,BoundingBox
        centro_y = [];%ê�����ĵ�
        for i = 1:length(STATS)
            centro_y = [centro_y round(STATS(i).Centroid(2))];
        end
        [val,i_max]=max(centro_y);
        [val,i_min]=min(centro_y);
        start_y = STATS(i_min).BoundingBox(2)-STATS(i_min).BoundingBox(4)*0.5;
        end_y = STATS(i_max).BoundingBox(2) + STATS(i_max).BoundingBox(4)*1.5;
        %���ָ���
        se = strel('square',3);%�ṹԪ�أ����ڸ�ʴ���ͻ򿪱����㣩
        e_bw_im=imerode(bw_im,se);%��ʴ
        e_bw_im = bwareaopen(e_bw_im,100);
        se = strel('square',11);%�ṹԪ�أ����ڸ�ʴ���ͻ򿪱����㣩
        e_bw_im=imdilate(e_bw_im,se);%����

        dy = (end_y-start_y)/length(STATS);
        %׼��֤��
        kh_start_x = int32(935.0/2530*w);
        kh_end_x = int32(2225.0/2530*w);
        kh_start_y = int32(start_y+dy*5);
        kh_end_y = int32(start_y+dy*15);
        %kh_area = e_bw_im(kh_start_y:kh_end_y,kh_start_x:kh_end_x);
        dx = (kh_end_x-kh_start_x)*1.0/9;
        kh_dy = (kh_end_y-kh_start_y)*0.1;
        %figure,imshow(kh_area),title('kh_area');
        %% ʶ�𿼺�
        thresh_optarea = dx*kh_dy/5;
        kh_num=[];
%         figure('name','boxes');
        for i=1:9
            cnt = 0;%�жϿ���һ�����˼���ѡ��
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
                        errno = 2;%����ĳ�ж�����
                        disp('����ĳ����Ϳ���󣬶���');
                        return;
                    end
                    kh_num = [kh_num j-1];
                end
            end
            if cnt==0
                errno = 3;%����ĳ��©����
                disp('����ĳ����Ϳ����©��');
                return;
            end
        end
        
        
        
        if SHOW
            merge = uint8(double(e_bw_im).*127.0+double(corrected_im_gray)*0.5);
            figure('name','��Ϳ���򻮷�'),imshow(e_bw_im),title('��Ϳ���򻮷�'),hold on;%e_bw_im ���� merge
            %������
            for i = 1:length(STATS)
                plot(x_start+STATS(i).Centroid(1),STATS(i).Centroid(2),'x','LineWidth',4,'Color','b');
                xy = [2 start_y+dy*(i-1); w-2 start_y+dy*(i-1)];
                plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            end
            xy = [2 start_y+dy*(i); w-2 start_y+dy*(i)];
            plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            %��������������
            for i = 1:10
                xy = [kh_start_x+(i-1)*dx start_y; kh_start_x+(i-1)*dx kh_end_y];
                plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','r');
            end
%             xy = [kh_start_x+(i)*kh_dx start_y; kh_start_x+(i)*kh_dx kh_end_y];
%             plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','b');
            
        end
        %% ʶ��ѡ��
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
        %ѡ��
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
        %% ѡ��ʶ��
        ANS = {'A','B','C','D','wrong'};
        for i=1:40
            opt_rlt(i) = get_rlt(opt_map{i});
        end
        %% ���ݴ���
        %����opt_rlt����ȷ�𰸵�Excel ���
        filename = strcat(rlt_excel_dir,'/',char(kh_num+'0'),'.xlsx');%����Ϊ�ļ���
        excel_gen = cell(42,4);
        excel_gen{1,1}='���';
        excel_gen{1,2}='��������';
        excel_gen{1,3}='��ȷ��';
        excel_gen{1,4}='�÷�';
        if  exist(answer_excel) ~= 0
            [num,txt,raw]=xlsread(answer_excel);
        else
            errno = 2;
            disp('û���ҵ���׼��Excel���');
            return ;
        end
        score = 0;
        count = 0;%��ȷ����
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
%         excel_gen{len+2,1}='����';
        excel_gen{len+2,1}=strcat('����: ',char(kh_num+'0'));
        excel_gen{len+2,3}='�ܷ�';
        excel_gen{len+2,4}=score;
        %����������Excel
        if  exist(rlt_excel_dir)==0
            mkdir(rlt_excel_dir);
        end
        
        xlswrite(filename,excel_gen);
        %uiwait;
        
    end  
end